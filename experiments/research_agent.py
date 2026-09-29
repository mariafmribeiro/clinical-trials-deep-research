import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from gpt_researcher import GPTResearcher

load_dotenv()


def get_search_url() -> str:
    return (
        os.getenv("RETRIEVER_ENDPOINT")
        or os.getenv("CUSTOM_RETRIEVER_URL")
        or os.getenv("SEARCH_API_URL")
        or os.getenv("SEARCH_URL")
        or ""
    )


def setup_retriever_env() -> str:
    search_url = get_search_url()

    if not search_url:
        raise RuntimeError(
            "Retriever endpoint is not configured. Export RETRIEVER_ENDPOINT "
            "before starting the researcher."
        )

    os.environ["RETRIEVER_ENDPOINT"] = search_url
    os.environ["CUSTOM_RETRIEVER_URL"] = search_url
    os.environ["SEARCH_API_URL"] = search_url
    os.environ["SEARCH_URL"] = search_url
    os.environ["RETRIEVER"] = "custom"
    os.environ["SEARCH_RETRIEVER"] = "custom"

    return search_url


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def context_to_text(context: Any) -> str:
    if context is None:
        return ""

    if isinstance(context, list):
        return "\n\n" + ("\n\n" + ("=" * 80) + "\n\n").join(str(item) for item in context)

    return str(context)


def build_input_metadata(
    query: str,
    pmid: str | None,
    title: str | None,
    search_url: str,
) -> dict[str, Any]:
    return {
        "pmid": pmid,
        "title": title,
        "query": query,
        "search_url": search_url,
        "openai_api_base": os.getenv("OPENAI_API_BASE"),
        "model_path": os.getenv("MODEL_PATH"),
        "served_model_name": os.getenv("SERVED_MODEL_NAME"),
        "request_model_name": os.getenv("REQUEST_MODEL_NAME"),
        "lora_adapter_path": os.getenv("LORA_ADAPTER_PATH") or None,
        "lora_adapter_name": os.getenv("LORA_ADAPTER_NAME") or None,
        "retriever": os.getenv("RETRIEVER"),
        "search_retriever": os.getenv("SEARCH_RETRIEVER"),
        "report_type": "deep",
        "report_source": "custom",
    }


def build_research_artifacts(researcher: GPTResearcher) -> dict[str, Any]:
    context = researcher.get_research_context()
    source_urls = researcher.get_source_urls()

    if isinstance(context, list):
        context_items = len(context)
        context_chars = sum(len(str(item)) for item in context)
    else:
        context_items = 1 if context else 0
        context_chars = len(str(context)) if context else 0

    return {
        "context": context,
        "source_urls": source_urls,
        "costs": researcher.get_costs(),
        "step_costs": researcher.get_step_costs(),
        "context_items": context_items,
        "context_chars": context_chars,
    }


async def run_research(
    query: str,
    output_dir: str,
    pmid: str | None = None,
    title: str | None = None,
    skip_report: bool = False,
) -> None:
    print("--- Starting Research ---")
    print(f"PMID: {pmid}")
    print(f"Title: {title}")
    print(f"Query: {query}")

    search_url = setup_retriever_env()

    print(f"--- Base URL: {os.getenv('OPENAI_API_BASE')}")
    print(f"--- Search URL: {search_url}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_metadata = build_input_metadata(
        query=query,
        pmid=pmid,
        title=title,
        search_url=search_url,
    )
    write_json(output_path / "input.json", input_metadata)

    try:
        researcher = GPTResearcher(
            query=query,
            report_type="deep",
            report_source="custom",
            source_urls=None,
            verbose=True,
        )

        print("--- Step 1: Conducting Research... ---")
        await researcher.conduct_research()

        artifacts = build_research_artifacts(researcher)
        write_json(output_path / "research_artifacts.json", artifacts)
        write_json(output_path / "source_urls.json", artifacts["source_urls"])
        write_text(output_path / "research_context.txt", context_to_text(artifacts["context"]))
        query_trace = getattr(
            getattr(researcher, "deep_researcher", None),
            "query_trace",
            None,
        )
        if query_trace:
            write_json(output_path / "query_generation_trace.json", query_trace)

        print(f"--- Retrieved context items: {artifacts['context_items']}")
        print(f"--- Retrieved source URLs: {len(artifacts['source_urls'])}")
        print(f"--- Research costs: ${artifacts['costs']:.6f}")

        if skip_report:
            run_summary = {
                "pmid": pmid,
                "title": title,
                "query": query,
                "search_url": search_url,
                "report_skipped": True,
                "report_generated": False,
                "report_length": 0,
                "context_items": artifacts["context_items"],
                "context_chars": artifacts["context_chars"],
                "source_url_count": len(artifacts["source_urls"]),
                "costs": artifacts["costs"],
                "step_costs": artifacts["step_costs"],
            }
            write_json(output_path / "run_summary.json", run_summary)
            print("--- Step 2: Skipping report generation. ---")
            return

        print("--- Step 2: Writing Report... ---")
        report = await researcher.write_report()

        run_summary = {
            "pmid": pmid,
            "title": title,
            "query": query,
            "search_url": search_url,
            "report_skipped": False,
            "report_generated": bool(report and len(report) > 10),
            "report_length": len(report) if report else 0,
            "context_items": artifacts["context_items"],
            "context_chars": artifacts["context_chars"],
            "source_url_count": len(artifacts["source_urls"]),
            "costs": artifacts["costs"],
            "step_costs": artifacts["step_costs"],
        }

        if report and len(report) > 10:
            report_file = output_path / "generated_report.md"
            write_text(report_file, report)
            print(f"--- SUCCESS! Report saved to {report_file} ({len(report)} characters) ---")
        else:
            print("--- WARNING: Report generated was empty or too short. ---")

        write_json(output_path / "run_summary.json", run_summary)

    except Exception as e:
        print(f"--- CRITICAL ERROR during research: {str(e)} ---")
        raise


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Research query to run.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where outputs will be saved.",
    )

    parser.add_argument(
        "--pmid",
        type=str,
        default=None,
        help="Optional PMID associated with this query.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional review title associated with this query.",
    )

    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Run research only and skip final report generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_research(
            query=args.query,
            output_dir=args.output_dir,
            pmid=args.pmid,
            title=args.title,
            skip_report=args.skip_report,
        )
    )
