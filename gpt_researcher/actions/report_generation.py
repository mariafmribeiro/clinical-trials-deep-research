import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from urllib import request
from ..config.config import Config
from ..utils.llm import create_chat_completion
from ..utils.logger import get_formatted_logger
from ..prompts import PromptFamily, get_prompt_by_report_type
from ..utils.enum import Tone

logger = get_formatted_logger()

NCT_ID_PATTERN = re.compile(r"NCT\d{8}", re.IGNORECASE)


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


async def _save_report_prompt_audit(
    messages: list[dict[str, str]],
    model: str,
    requested_max_tokens: int | None,
) -> None:
    """Record exactly which NCT IDs remain after vLLM prompt truncation."""
    if os.getenv("SAVE_REPORT_PROMPT_AUDIT", "1") != "1":
        return

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or ""
    if not base_url or not ("127.0.0.1" in base_url or "localhost" in base_url):
        return

    server_root = base_url.rstrip("/")
    if server_root.endswith("/v1"):
        server_root = server_root[:-3]

    try:
        tokenized = await asyncio.to_thread(
            _post_json,
            f"{server_root}/tokenize",
            {
                "model": model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        token_ids = list(tokenized.get("tokens") or [])

        output_margin = int(os.getenv("LLM_OUTPUT_TOKEN_MARGIN", "128"))
        sent_max_tokens = None
        if requested_max_tokens is not None:
            sent_max_tokens = max(64, int(requested_max_tokens) - output_margin)

        context_window = int(os.getenv("VLLM_CONTEXT_WINDOW", "16384"))
        context_buffer = int(os.getenv("VLLM_CONTEXT_BUFFER", "512"))
        safe_prompt_tokens = context_window
        if sent_max_tokens is not None:
            safe_prompt_tokens -= sent_max_tokens + context_buffer
        safe_prompt_tokens = max(1, safe_prompt_tokens)

        was_truncated = len(token_ids) > safe_prompt_tokens
        visible_token_ids = token_ids[-safe_prompt_tokens:] if was_truncated else token_ids
        detokenized = await asyncio.to_thread(
            _post_json,
            f"{server_root}/detokenize",
            {"model": model, "tokens": visible_token_ids},
        )
        visible_prompt = str(detokenized.get("prompt") or "")
        full_text = "\n".join(str(message.get("content") or "") for message in messages)
        full_nct_ids = sorted({value.upper() for value in NCT_ID_PATTERN.findall(full_text)})
        visible_nct_ids = sorted({value.upper() for value in NCT_ID_PATTERN.findall(visible_prompt)})

        output_dir = Path(
            os.getenv("FINAL_CONTEXT_DEBUG_DIR", str(Path("logs") / "final_context_debug"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        label = os.getenv("FINAL_CONTEXT_DEBUG_LABEL", "research")
        audit = {
            "model": model,
            "context_window": context_window,
            "context_buffer": context_buffer,
            "requested_max_tokens": requested_max_tokens,
            "sent_max_tokens": sent_max_tokens,
            "full_prompt_token_count": len(token_ids),
            "safe_prompt_token_count": safe_prompt_tokens,
            "visible_prompt_token_count": len(visible_token_ids),
            "was_truncated": was_truncated,
            "truncated_token_count": max(0, len(token_ids) - len(visible_token_ids)),
            "full_prompt_nct_ids": full_nct_ids,
            "visible_prompt_nct_ids": visible_nct_ids,
            "lost_prompt_nct_ids": sorted(set(full_nct_ids) - set(visible_nct_ids)),
        }
        (output_dir / f"{label}_report_prompt_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / f"{label}_visible_report_prompt.txt").write_text(
            visible_prompt,
            encoding="utf-8",
        )
        logger.info(
            "Report prompt audit: %d/%d tokens visible; truncated=%s; NCT IDs=%d/%d",
            len(visible_token_ids),
            len(token_ids),
            was_truncated,
            len(visible_nct_ids),
            len(full_nct_ids),
        )
    except Exception as exc:
        logger.warning("Could not save report prompt audit: %s", exc)


async def write_report_introduction(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> str:
    """
    Generate an introduction for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        prompt_family: Family of prompts

    Returns:
        str: The generated introduction.
    """
    try:
        introduction = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": prompt_family.generate_report_introduction(
                    question=query,
                    research_summary=context,
                    language=config.language
                )},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return introduction
    except Exception as e:
        logger.error(f"Error in generating report introduction: {e}")
    return ""


async def write_conclusion(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> str:
    """
    Write a conclusion for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        prompt_family: Family of prompts

    Returns:
        str: The generated conclusion.
    """
    try:
        conclusion = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {
                    "role": "user",
                    "content": prompt_family.generate_report_conclusion(query=query,
                                                                        report_content=context,
                                                                        language=config.language),
                },
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return conclusion
    except Exception as e:
        logger.error(f"Error in writing conclusion: {e}")
    return ""


async def summarize_url(
    url: str,
    content: str,
    role: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    **kwargs
) -> str:
    """
    Summarize the content of a URL.

    Args:
        url (str): The URL to summarize.
        content (str): The content of the URL.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.

    Returns:
        str: The summarized content.
    """
    try:
        summary = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"Summarize the following content from {url}:\n\n{content}"},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return summary
    except Exception as e:
        logger.error(f"Error in summarizing URL: {e}")
    return ""


async def generate_draft_section_titles(
    query: str,
    current_subtopic: str,
    context: str,
    role: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> List[str]:
    """
    Generate draft section titles for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        prompt_family: Family of prompts

    Returns:
        List[str]: A list of generated section titles.
    """
    try:
        section_titles = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": prompt_family.generate_draft_titles_prompt(
                    current_subtopic, query, context)},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=None,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
        return section_titles.split("\n")
    except Exception as e:
        logger.error(f"Error in generating draft section titles: {e}")
    return []


async def generate_report(
    query: str,
    context,
    agent_role_prompt: str,
    report_type: str,
    tone: Tone,
    report_source: str,
    websocket,
    cfg,
    main_topic: str = "",
    existing_headers: list = [],
    relevant_written_contents: list = [],
    cost_callback: callable = None,
    custom_prompt: str = "", # This can be any prompt the user chooses with the context
    headers=None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    available_images: list = None,
    **kwargs
):
    """
    generates the final report
    Args:
        query:
        context:
        agent_role_prompt:
        report_type:
        websocket:
        tone:
        cfg:
        main_topic:
        existing_headers:
        relevant_written_contents:
        cost_callback:
        prompt_family: Family of prompts
        available_images: Pre-generated images to embed in the report

    Returns:
        report:

    """
    available_images = available_images or []
    generate_prompt = get_prompt_by_report_type(report_type, prompt_family)
    report = ""

    if report_type == "subtopic_report":
        content = f"{generate_prompt(query, existing_headers, relevant_written_contents, main_topic, context, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words, language=cfg.language)}"
    elif custom_prompt:
        content = f"{custom_prompt}\n\nContext: {context}"
    else:
        content = f"{generate_prompt(query, context, report_source, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words, language=cfg.language)}"
    
    # Add available images instruction if images were pre-generated
    if available_images:
        images_info = "\n".join([
            f"- Image {i+1}: ![{img.get('title', img.get('alt_text', 'Illustration'))}]({img['url']}) - {img.get('section_hint', 'General')}"
            for i, img in enumerate(available_images)
        ])
        content += f"""

AVAILABLE IMAGES:
You have the following pre-generated images available. Embed them in relevant sections of your report using the exact markdown syntax provided:

{images_info}

Place each image on its own line after the relevant section header or paragraph. Use all available images where they add value to the content."""
    report_messages = [
        {"role": "system", "content": f"{agent_role_prompt}"},
        {"role": "user", "content": content},
    ]
    await _save_report_prompt_audit(
        messages=report_messages,
        model=cfg.smart_llm_model,
        requested_max_tokens=cfg.smart_token_limit,
    )

    try:
        report = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=report_messages,
            temperature=0.1,
            llm_provider=cfg.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=cfg.smart_token_limit,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )
    except Exception:
        try:
            report = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[
                    {"role": "user", "content": f"{agent_role_prompt}\n\n{content}"},
                ],
                temperature=0.35,
                llm_provider=cfg.smart_llm_provider,
                stream=True,
                websocket=websocket,
                max_tokens=cfg.smart_token_limit,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs
            )
        except Exception as e:
            print(f"Error in generate_report: {e}")

    return report
