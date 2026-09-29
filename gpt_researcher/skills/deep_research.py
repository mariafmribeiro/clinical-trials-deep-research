from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
import json
import os
from pathlib import Path


import json_repair

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource, Tone
from ..actions.query_processing import get_search_results

logger = logging.getLogger(__name__)

# Maximum words allowed in context (25k words for safety margin)
MAX_CONTEXT_WORDS = 25000

JSON_BLOCK_PATTERNS = [
    re.compile(
        r"```(?:json)?\s*(?P<payload>[\s\S]*?)```",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<payload>\[[\s\S]*\])"),
    re.compile(r"(?P<payload>\{[\s\S]*\})"),
]

QUERY_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*Query:\s*(?P<query>.+)$",
    re.IGNORECASE,
)
GOAL_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*(?:Goal|Research Goal):\s*(?P<goal>.+)$",
    re.IGNORECASE,
)
QUESTION_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*(?:Question:\s*)?(?P<question>.+\?)$",
    re.IGNORECASE,
)
LEARNING_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*Learning(?:\s*\[(?P<citation>[^\]]+)\])?:\s*(?P<learning>.+)$",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s\]\)>\",;]+")
NCT_ID_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def join_context_items(context_list: List[str]) -> str:
    return "\n\n---\n\n".join(str(item) for item in context_list if item)


def extract_nct_ids_from_text(text: str) -> List[str]:
    seen = set()
    ordered = []
    for match in NCT_ID_PATTERN.findall(text or ""):
        value = match.upper()
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _extract_json_payloads(response: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for pattern in JSON_BLOCK_PATTERNS:
        for match in pattern.finditer(response):
            candidate = match.group("payload").strip()
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    return candidates


def _load_repaired_json(response: str) -> Any:
    for candidate in [response.strip(), *_extract_json_payloads(response)]:
        if not candidate:
            continue
        try:
            return json_repair.loads(candidate)
        except Exception as exc:
            logger.debug(
                "json_repair failed on candidate (%d chars): %s",
                len(candidate), exc,
            )
            continue
    return None


def parse_search_queries_response(response: str, num_queries: int) -> List[Dict[str, str]]:
    parsed = _load_repaired_json(response)
    candidate_queries = parsed
    if isinstance(parsed, dict):
        candidate_queries = parsed.get("queries") or parsed.get("searchQueries") or parsed.get("items")

    if isinstance(candidate_queries, list):
        queries = [
            {
                "query": item["query"].strip(),
                "researchGoal": item["researchGoal"].strip(),
            }
            for item in candidate_queries
            if isinstance(item, dict) and item.get("query") and item.get("researchGoal")
        ]
        if queries:
            return queries[:num_queries]

    queries: List[Dict[str, str]] = []
    current_query: Dict[str, str] = {}

    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        query_match = QUERY_LINE_PATTERN.match(line)
        goal_match = GOAL_LINE_PATTERN.match(line)

        if query_match:
            if current_query.get("query") and current_query.get("researchGoal"):
                queries.append(current_query)
            current_query = {"query": query_match.group("query").strip()}
        elif goal_match and current_query.get("query"):
            current_query["researchGoal"] = goal_match.group("goal").strip()

    if current_query.get("query") and current_query.get("researchGoal"):
        queries.append(current_query)

    # -----------------DEBUG ------------------------


    print("\n========== PARSED DEEP_RESEARCH SEARCH QUERIES ==========", flush=True)
    print(queries, flush=True)
    print(f"Number parsed: {len(queries) if queries else 0}", flush=True)
    print("==========================================================\n", flush=True)

    # ------------------------------------------------------

    return queries[:num_queries]


def parse_follow_up_questions_response(response: str, num_questions: int) -> List[str]:
    parsed = _load_repaired_json(response)
    candidate_questions = parsed
    if isinstance(parsed, dict):
        candidate_questions = parsed.get("questions") or parsed.get("followUpQuestions") or parsed.get("items")

    if isinstance(candidate_questions, list):
        questions = [str(item).strip() for item in candidate_questions if str(item).strip()]
        if questions:
            return questions[:num_questions]

    questions: List[str] = []
    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        question_match = QUESTION_LINE_PATTERN.match(line)
        if question_match:
            questions.append(question_match.group("question").strip())

    return questions[:num_questions]


def parse_research_results_response(response: str, num_learnings: int) -> Dict[str, Any]:
    parsed = _load_repaired_json(response)

    if isinstance(parsed, dict):
        learnings_payload = parsed.get("learnings", [])
        follow_up_payload = parsed.get("followUpQuestions") or parsed.get("questions") or []
        learnings: List[str] = []
        citations: Dict[str, str] = {}

        if isinstance(learnings_payload, list):
            for item in learnings_payload:
                if isinstance(item, dict):
                    learning = str(item.get("insight") or item.get("learning") or "").strip()
                    citation = str(item.get("sourceUrl") or item.get("citation") or "").strip()
                else:
                    learning = str(item).strip()
                    citation = ""

                if learning:
                    learnings.append(learning)
                    if citation:
                        citations[learning] = citation

        questions = [str(item).strip() for item in follow_up_payload if str(item).strip()]
        if learnings or questions:
            return {
                "learnings": learnings[:num_learnings],
                "followUpQuestions": questions[:num_learnings],
                "citations": citations,
            }

    learnings: List[str] = []
    questions: List[str] = []
    citations: Dict[str, str] = {}

    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        learning_match = LEARNING_LINE_PATTERN.match(line)
        question_match = QUESTION_LINE_PATTERN.match(line)

        if learning_match:
            learning = learning_match.group("learning").strip()
            citation = (learning_match.group("citation") or "").strip()
            if not citation:
                url_match = URL_PATTERN.search(learning)
                if url_match:
                    citation = url_match.group(0)
                    learning = learning.replace(citation, "").strip(" -")
            if learning:
                learnings.append(learning)
                if citation:
                    citations[learning] = citation
        elif question_match:
            questions.append(question_match.group("question").strip())

    return {
        "learnings": learnings[:num_learnings],
        "followUpQuestions": questions[:num_learnings],
        "citations": citations,
    }

def count_words(text) -> int:
    """Count words in a text string. Handles both strings and lists."""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    return len(str(text).split())

def interleave_trial_context_blocks(context_list: List[str]) -> List[str]:
    """Interleave trial blocks by within-breadth rank.

    This prevents one complete breadth group from consuming the remaining
    context budget before trials from other breadths are considered.
    """
    breadth_blocks = [
        _split_trial_context_blocks(context_item)
        for context_item in context_list
        if str(context_item or "").strip()
    ]

    if not breadth_blocks:
        return []

    ordered_blocks = []
    max_blocks = max(len(blocks) for blocks in breadth_blocks)

    for rank_index in range(max_blocks):
        for blocks in breadth_blocks:
            if rank_index < len(blocks):
                ordered_blocks.append(blocks[rank_index])

    return ordered_blocks


def pack_context_from_start(
    context_list: List[str],
    max_words: int,
) -> List[str]:
    """Fill the available budget without stopping at the first non-fitting item."""
    total_words = 0
    kept = []

    for item in context_list:
        words = count_words(item)

        if total_words + words > max_words:
            continue

        kept.append(item)
        total_words += words

    return kept

def trim_context_to_word_limit(context_list: List[str], max_words: int = MAX_CONTEXT_WORDS) -> List[str]:
    """Trim context list to stay within word limit while preserving most recent/relevant items"""
    total_words = 0
    trimmed_context = []

    # Process in reverse to keep most recent items
    for item in reversed(context_list):
        words = count_words(item)
        if total_words + words <= max_words:
            trimmed_context.insert(0, item)  # Insert at start to maintain original order
            total_words += words
        else:
            break

    return trimmed_context

def trim_context_from_start(
    context_list: List[str],
    max_words: int,
) -> List[str]:
    """Keep items from the beginning while respecting a word budget."""
    total_words = 0
    kept = []

    for item in context_list:
        words = count_words(item)

        if total_words + words > max_words:
            break

        kept.append(item)
        total_words += words

    return kept


TRIAL_BLOCK_START_PATTERN = re.compile(
    r"(?=^### Trial\s+\d+:\s*NCT\d{8}\b)",
    re.IGNORECASE | re.MULTILINE,
)


def trial_level_global_dedupe_enabled() -> bool:
    value = os.getenv("TRIAL_LEVEL_CONTEXT_GLOBAL_DEDUPE", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _split_trial_context_blocks(context_item: str) -> List[str]:
    text = str(context_item or "").strip()
    if not text:
        return []
    if not TRIAL_BLOCK_START_PATTERN.search(text):
        return [text]
    return [part.strip() for part in TRIAL_BLOCK_START_PATTERN.split(text) if part.strip()]


def dedupe_trial_level_context_blocks(context_list: List[str]) -> List[str]:
    """Interleave breadth ranks, then keep the first block for each NCT ID."""
    ordered_blocks = interleave_trial_context_blocks(context_list or [])
    seen_ncts = set()
    deduped_blocks: List[str] = []

    for block in ordered_blocks:
        match = NCT_ID_PATTERN.search(block)
        if not match:
            deduped_blocks.append(block)
            continue

        nct_id = match.group(0).upper()
        if nct_id in seen_ncts:
            continue

        seen_ncts.add(nct_id)
        deduped_blocks.append(block)

    return deduped_blocks


def trial_level_learning_dedupe_enabled() -> bool:
    value = os.getenv("TRIAL_LEVEL_CONTEXT_DEDUPE_WITH_LEARNINGS", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _has_clinical_results_excerpt(trial_block: str) -> bool:
    return bool(
        re.search(
            r"^(?:Clinical results excerpt|Posted outcome results|Posted safety results):\s*\S",
            trial_block or "",
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )


def select_trial_context_fallback(
    trial_context: List[str],
    learning_context: List[str],
) -> List[str]:
    """Keep a small, globally ranked trial-card fallback after learnings.

    Cards already represented by a learning are redundant. The exception is a
    card with explicit registry results, which can preserve primary evidence
    that the generated learning may have shortened or omitted.
    """
    fallback_k = max(0, int(os.getenv("TRIAL_LEVEL_CONTEXT_GLOBAL_FALLBACK_K", "10")))
    result_evidence_k = max(
        0,
        int(os.getenv("TRIAL_LEVEL_CONTEXT_RESULT_EVIDENCE_K", "5")),
    )
    keep_result_evidence = os.getenv(
        "TRIAL_LEVEL_CONTEXT_KEEP_RESULT_EVIDENCE",
        "1",
    ).strip().lower() in {"1", "true", "yes", "on"}

    learning_ncts = {
        nct_id
        for learning in learning_context
        for nct_id in extract_nct_ids_from_text(str(learning))
    }

    # This also splits breadth-level strings, interleaves their ranks, and
    # preserves the first selected representation of each NCT.
    trial_blocks = dedupe_trial_level_context_blocks(trial_context)
    non_trial_blocks: List[str] = []
    missing_from_learnings: List[tuple[int, str]] = []
    result_evidence: List[tuple[int, str]] = []

    for order, block in enumerate(trial_blocks):
        block_ncts = extract_nct_ids_from_text(str(block))
        if not block_ncts:
            non_trial_blocks.append(block)
            continue

        nct_id = block_ncts[0]
        if nct_id not in learning_ncts:
            missing_from_learnings.append((order, block))
        elif keep_result_evidence and _has_clinical_results_excerpt(block):
            result_evidence.append((order, block))

    # Trial blocks are ordered strongest-first after rank interleaving.
    selected = (
        result_evidence[:result_evidence_k] if result_evidence_k else []
    ) + (
        missing_from_learnings[:fallback_k] if fallback_k else []
    )
    selected.sort(key=lambda item: item[0])
    selected_blocks = non_trial_blocks + [block for _, block in selected]

    logger.info(
        "Learning/card dedupe kept %d result-evidence cards and %d fallback cards "
        "from %d globally deduplicated trial blocks (%d NCTs already in learnings)",
        min(len(result_evidence), result_evidence_k),
        min(len(missing_from_learnings), fallback_k),
        len(trial_blocks),
        len(learning_ncts),
    )
    return selected_blocks

# ------------ HELPER FUNCTIONS -------------
def dedupe_search_query_objects(items: List[Dict[str, str]], max_items: int | None = None) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen_queries = set()

    for item in items or []:
        if not isinstance(item, dict):
            continue

        query = item.get("query")
        research_goal = item.get("researchGoal")

        if not query or not research_goal:
            continue

        if query in seen_queries:
            continue

        seen_queries.add(query)
        deduped.append({
            "query": query,
            "researchGoal": research_goal,
        })

    if max_items is not None:
        return deduped[:max_items]
    return deduped


def dedupe_strings(items: List[str], max_items: int | None = None) -> List[str]:
    deduped: List[str] = []
    seen = set()

    for item in items or []:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    if max_items is not None:
        return deduped[:max_items]
    return deduped


def merge_citation_maps(target: Dict[str, str], incoming: Dict[str, str]) -> None:
    """Merge semicolon-delimited sources without losing duplicate learnings."""
    for learning, citation in (incoming or {}).items():
        citation = str(citation or "").strip()
        if not citation:
            continue

        existing = str(target.get(learning, "") or "").strip()
        sources = []
        seen_sources = set()
        for value in (existing, citation):
            for source in value.split(";"):
                source = source.strip()
                if source and source not in seen_sources:
                    seen_sources.add(source)
                    sources.append(source)
        target[learning] = "; ".join(sources)


def build_combined_query(original_query: str, search_directions: List[str]) -> str:
    parts = [original_query]
    parts.extend(dedupe_strings(search_directions))
    parts = [p for p in parts if p]

    if not parts:
        return original_query

    return " | ".join(parts)


def build_recursive_query(research_goal: str, follow_up_questions: List[str]) -> str:
    parts = [research_goal]
    parts.extend(dedupe_strings(follow_up_questions, max_items=3))
    parts = [p for p in parts if p]

    if not parts:
        return research_goal

    return " | ".join(parts)

# ---------------------------------------------------

class ResearchProgress:
    def __init__(self, total_depth: int, total_breadth: int):
        self.current_depth = 1  # Start from 1 and increment up to total_depth
        self.total_depth = total_depth
        self.current_breadth = 0  # Start from 0 and count up to total_breadth as queries complete
        self.total_breadth = total_breadth
        self.current_query: Optional[str] = None
        self.total_queries = 0
        self.completed_queries = 0


class DeepResearchSkill:
    def __init__(self, researcher):
        self.researcher = researcher
        self.breadth = getattr(researcher.cfg, 'deep_research_breadth', 4)
        self.depth = getattr(researcher.cfg, 'deep_research_depth', 2)
        self.concurrency_limit = getattr(researcher.cfg, 'deep_research_concurrency', 2)
        self.websocket = researcher.websocket
        self.tone = researcher.tone
        self.config_path = researcher.cfg.config_path if hasattr(researcher.cfg, 'config_path') else None
        self.headers = researcher.headers or {}
        self.visited_urls = researcher.visited_urls
        self.learnings = []
        self.research_sources = []  # Track all research sources
        self.context = []  # Track all context
        self.query_trace = {
            "root_query": researcher.query,
            "search_directions": [],
            "combined_query": None,
            "runs": [],
        }

    def _save_final_context_debug(self, context_with_citations: List[str], final_context: List[str], results: Dict[str, Any]) -> None:
        generated_id = self.researcher._generate_research_id() if hasattr(self.researcher, "_generate_research_id") else f"research_{int(time.time())}"
        research_id = os.getenv("FINAL_CONTEXT_DEBUG_LABEL") or generated_id
        output_dir = Path(os.getenv("FINAL_CONTEXT_DEBUG_DIR", str(Path("logs") / "final_context_debug")))
        output_dir.mkdir(parents=True, exist_ok=True)

        pre_trim_text = join_context_items(context_with_citations)
        post_trim_text = join_context_items(final_context)

        debug_payload = {
            "research_id": research_id,
            "generated_research_id": generated_id,
            "debug_output_dir": str(output_dir),
            "root_query": self.researcher.query,
            "breadth": self.breadth,
            "depth": self.depth,
            "pre_trim_item_count": len(context_with_citations),
            "post_trim_item_count": len(final_context),
            "pre_trim_word_count": count_words(pre_trim_text),
            "post_trim_word_count": count_words(post_trim_text),
            "pre_trim_nct_ids": extract_nct_ids_from_text(pre_trim_text),
            "post_trim_nct_ids": extract_nct_ids_from_text(post_trim_text),
            "visited_urls": results.get("visited_urls", []),
            "visited_url_count": len(results.get("visited_urls", [])),
        }

        (output_dir / f"{research_id}_final_context_debug.json").write_text(
            json.dumps(debug_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / f"{research_id}_pre_trim_context.txt").write_text(pre_trim_text, encoding="utf-8")
        (output_dir / f"{research_id}_post_trim_context.txt").write_text(post_trim_text, encoding="utf-8")

    async def generate_search_queries(self, query: str, num_queries: int = 3) -> List[Dict[str, str]]:
        """Generate compact registry-style search queries for local Whoosh retrieval."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher generating search queries for a local ClinicalTrials.gov index. "
                    "Return valid JSON only. Do not include markdown, code fences, bullets, numbering, or prose."
                ),
            },
# --------------- PERMISSIVE PROMPT ------------------------------------------            
{
"role": "user",
"content": f"""
Generate exactly {num_queries} short, complementary BM25 queries for retrieving ClinicalTrials.gov trial records relevant to this systematic review question.

Keep one query close to the original wording.

For the other queries, expand the terminology using useful:
- condition or population synonyms and clinically relevant umbrella conditions;
- intervention synonyms, concrete class members, older names, development codes, or trial acronyms.

Each query must contain a relevant condition or population and an intervention term.

Each query should add meaningful new retrieval terminology. Abbreviations may be used in one query as an additional alternative. Avoid generic wording, minor rephrasings, outcomes, durations, comparators, and Boolean operators.

Return only a JSON array with exactly {num_queries} objects:
[{{"query": "<query>", "researchGoal": "<brief purpose>"}}]

Question:
{query}
"""
},

# -------------------- PICO STYLE PROMPT ------------------------------------
# {
# "role": "user",
# "content": f"""
# Before generating queries, identify the review's PICO: Population/condition, Intervention, Comparator (only if central to trial identification), Outcomes (only if they disambiguate the topic). Do not output the PICO, use it only to write better queries.

# Generate exactly {num_queries} short, complementary BM25 queries for retrieving ClinicalTrials.gov trial records relevant to this systematic review question, grounded in that PICO breakdown.

# Keep one query close to the original wording.

# For the other queries, expand the terminology using useful:
# - condition or population synonyms and clinically relevant umbrella conditions (from the Population element);
# - intervention synonyms, concrete class members, older names, development codes, or trial acronyms (from the Intervention element).

# Each query must contain a relevant condition/population and an intervention term.

# Return only a JSON array with exactly {num_queries} objects:
# [{{"query": "<query>", "researchGoal": "<brief purpose>"}}]

# Question:
# {query}
# """
# },

# --------------------------------------------------------------------------------


# ------ WEIRD PROMPT --------------------------------------------------------------



# ----------------------------------------------------------------------------------------



        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=self.researcher.cfg.reasoning_effort,
            temperature=0.1
        )

        print("\n========== RAW DEEP_RESEARCH QUERY GENERATION OUTPUT ==========", flush=True)
        print(response, flush=True)
        print("================================================================\n", flush=True)

        parsed_queries = parse_search_queries_response(response, num_queries * 2)
        parsed_queries = dedupe_search_query_objects(parsed_queries, max_items=num_queries)

        if not parsed_queries:
            parsed_queries = [{
                "query": query,
                "researchGoal": f"Retrieve trial records directly relevant to: {query}",
            }]

        print("\n========== DEDUPED DEEP_RESEARCH SEARCH QUERIES ==========", flush=True)
        print(parsed_queries, flush=True)
        print("==========================================================\n", flush=True)

        return parsed_queries[:num_queries]

    async def generate_research_plan(self, query: str, num_questions: int = 2) -> List[str]:
        """Generate compact search directions to diversify retrieval."""
        all_search_results = []

        for retriever in self.researcher.retrievers:
            try:
                results = await get_search_results(
                    query,
                    retriever,
                    researcher=self.researcher
                )
                all_search_results.extend(results)
            except Exception as e:
                logger.warning(f"Error with retriever {retriever.__name__}: {e}")

        # search_results = all_search_results
        # logger.info(f"Initial knowledge obtained: {len(search_results)} results")
        logger.info(f"Initial knowledge obtained: {len(all_search_results)} results")

        plan_results_limit = int(os.getenv("PLAN_RESEARCH_CONTEXT_RESULTS", "15"))
        plan_content_chars = int(os.getenv("PLAN_RESEARCH_CONTEXT_CHARS", "2000"))

        search_results = [
            {
                "nct_id": result.get("nct_id"),
                "title": result.get("title"),
                "raw_content": str(result.get("raw_content") or "")[:plan_content_chars],
            }
            for result in all_search_results[:plan_results_limit]
            if isinstance(result, dict)
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical trial registry search strategist. "
                    "Your task is to turn a medical research question into compact search directions "
                    "for a local ClinicalTrials.gov index. Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": f"""
    Original research question:
    {query}

    Generate compact registry retrieval directions, not natural-language questions.

Rules:
- Return compact registry search directions, not research questions.
- Each direction must preserve the core retrieval topic.
- In most cases, each direction must include the condition/population/context AND the intervention or intervention class.
- Do not return directions that contain only an outcome, only a comparator, only a population, or only a broad treatment concept.
- Directions should add alternative wording without changing the review topic.
- Useful alternatives include disease synonyms, older disease names, intervention synonyms, abbreviations, comparator wording, population subtype, or setting.
- Do not introduce interventions, comparators, populations, or outcomes not implied by the original question.
- No verbs like assess, evaluate, compare, determine.
- No question marks.
- No full sentences.

    Generate {num_questions} compact registry search directions.

    Return ONLY this JSON schema:
    {{"questions": ["<search direction 1>", "<search direction 2>"]}}

    Search results context:
    {search_results}
    """
            }
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.High.value,
            temperature=0.4,
            max_tokens=1000
        )

        # print("\n========== RAW RESEARCH PLAN LLM OUTPUT ==========", flush=True)
        # print(response, flush=True)
        # print("==================================================\n", flush=True)

        parsed_questions = parse_follow_up_questions_response(response, num_questions * 2)
        parsed_questions = dedupe_strings(parsed_questions, max_items=num_questions)

        # print("\n========== PARSED SEARCH DIRECTIONS ==========", flush=True)
        # print(parsed_questions, flush=True)
        # print(f"Number parsed: {len(parsed_questions) if parsed_questions else 0}", flush=True)
        # print("==============================================\n", flush=True)

        planned = parsed_questions[:num_questions]

        if query == self.researcher.query:
            self.query_trace["search_directions"] = planned

        return planned

        # return parsed_questions[:num_questions]

    async def process_research_results(self, query: str, context: str, num_learnings: int = 10) -> Dict[str, List[str]]:
        """Process research results to extract learnings and follow-up questions"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher analyzing search results. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user",
            #  "content": (
            #      f"Given the following research results for the query '{query}', extract key learnings and suggest "
            #      "follow-up questions. Extract trial-level learnings. Prefer one concise learning per relevant NCT ID. Always include the NCT ID in the insight and citation.\n\n"
            #      "Extract broad trial-level learnings across all relevant intervention classes, disease settings, comparators, and outcomes. Do not focus only on the first few trials or one drug if the query is broad.\n"
            #      "Return ONLY a JSON object using this exact schema:\n"
            #      '{"learnings": [{"insight": "<insight>", "sourceUrl": "<url or empty string>"}], '
            #      '"followUpQuestions": ["<question 1>", "<question 2>"]}\n\n'
            #      f"Research results:\n{context}"
            #  )}
    #         "content": (
    # f"Given the following retrieved ClinicalTrials.gov trial records for the query '{query}', "
    # f"extract as many distinct NCT-level learnings as possible, up to {num_learnings}.\n\n"

    # "Important rules:\n"
    # "- Do NOT collapse all trials into only 2 or 3 general statements.\n"
    # "- Extract one learning per relevant NCT ID whenever possible.\n"
    # "- If the record only describes design/outcomes and has no results, say 'registry design information only'.\n"
    # "- Preserve coverage across different drugs, intervention classes, comparators, populations, disease settings, and outcomes.\n"
    # "- If the query is broad, do not focus only on the first drug, first subgroup,  or first few trials.\n"
    # "- For each trial, extract: population/setting, intervention, comparator, outcomes measured, results if they are available.\n"
    # "- Do NOT claim effectiveness or safety unless the record explicitly contains results.\n"
    # "- The sourceUrl field MUST contain the NCT ID/source identifier for that learning.\n"
    # "- Do not merge multiple NCT IDs into one learning unless absolutely necessary.\n\n"

    # "Return ONLY a JSON object using this exact schema:\n"
    # "{\n"
    # '  "learnings": [\n'
    # '    {\n'
    # '      "insight": "NCT ID | population/setting | intervention vs comparator | outcomes/results | relevance to question",\n'
    # '      "sourceUrl": "NCT ID or source document name"\n'
    # "    }\n"
    # "  ],\n"
    # '  "followUpQuestions": []\n'
    # "}\n\n"
    # f"Research results:\n{context}"
    #     "content": (
    # f"Given the following ClinicalTrials.gov records retrieved for the research branch "
    # f"'{query}', inspect every distinct NCT ID and extract up to "
    # f"{num_learnings} trial-level learnings.\n\n"

    # "Selection rules:\n"
    # "- Evaluate every distinct NCT record before selecting the learnings.\n"
    # "- Include each trial that is directly or possibly relevant to this research branch.\n"
    # "- Do not select only representative examples or only the first records in the context.\n"
    # "- Preserve coverage across interventions, comparators, populations, study designs and outcomes.\n"
    # "- Missing clinical results or incomplete reporting are not reasons to exclude a relevant trial.\n"
    # "- If more relevant trials exist than the output limit, prioritize direct matches while "
    # "preserving coverage across the different evidence groups represented in the context.\n\n"

    # "Extraction rules:\n"
    # "- Produce exactly one learning per selected NCT ID.\n"
    # "- Never merge multiple NCT IDs into one learning.\n"
    # "- Include the exact NCT ID in both insight and sourceUrl.\n"
    # "- Do not claim effectiveness or safety unless explicit results are available.\n"
    # "- If no results are available, say 'registry design information only'.\n"
    # "- Keep each learning concise so that detail does not reduce trial coverage.\n\n"

    # "Return ONLY this JSON structure:\n"
    # "{\n"
    # '  "learnings": [\n'
    # "    {\n"
    # '      "insight": "NCT01234567 | population | intervention vs comparator | '
    # 'outcomes/results or registry design information only",\n'
    # '      "sourceUrl": "NCT01234567"\n'
    # "    }\n"
    # "  ],\n"
    # '  "followUpQuestions": []\n'
    # "}\n\n"
    # f"Trial records:\n{context}"
        "content": (
            f"Using the ClinicalTrials.gov records below, produce up to {num_learnings} "
            f"concise evidence syntheses relevant to this research branch:\n'{query}'\n\n"
            "Group trials when they provide related evidence about the same intervention, "
            "population, comparator, outcome, or study finding. Report effectiveness or safety "
            "only when explicit results are available; otherwise describe the available "
            "registry evidence. For each synthesis, list the supporting NCT IDs from the "
            "provided records in sourceUrl, separated by semicolons.\n\n"
            "Return valid JSON only:\n"
            "{\n"
            '  "learnings": [\n'
            '    {"insight": "<concise evidence synthesis>", '
            '"sourceUrl": "NCT########; NCT########"}\n'
            "  ],\n"
            '  "followUpQuestions": []\n'
            "}\n\n"
            f"ClinicalTrials.gov records:\n{context}"
        )}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.1,
            reasoning_effort=ReasoningEfforts.High.value,
            max_tokens=2000
        )

        

        return parse_research_results_response(response, num_learnings)

    async def deep_research(
            self,
            query: str,
            breadth: int,
            depth: int,
            learnings: List[str] = None,
            citations: Dict[str, str] = None,
            visited_urls: Set[str] = None,
            on_progress=None
    ) -> Dict[str, Any]:
        """Conduct deep iterative research"""
        print(f"\n📊 DEEP RESEARCH: depth={depth}, breadth={breadth}, query={query[:100]}...", flush=True)
        if learnings is None:
            learnings = []
        if citations is None:
            citations = {}
        if visited_urls is None:
            visited_urls = set()

        progress = ResearchProgress(depth, breadth)

        if on_progress:
            on_progress(progress)

        # Generate search queries
        print(f"🔎 Generating {breadth} search queries...", flush=True)
        serp_queries = await self.generate_search_queries(query, num_queries=breadth)
        serp_queries = dedupe_search_query_objects(serp_queries, max_items=breadth)

        run_record = {
            "input_query": query,
            "depth": depth,
            "breadth": breadth,
            "generated_queries": serp_queries,
            "results": [],
        }
        self.query_trace["runs"].append(run_record)


        print(f"✅ Generated {len(serp_queries)} queries: {[q['query'] for q in serp_queries]}", flush=True)
        progress.total_queries = len(serp_queries)

        all_learnings = learnings.copy()
        all_citations = citations.copy()
        all_visited_urls = visited_urls.copy()
        all_context = []
        all_sources = []

        # Process queries with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def process_query(serp_query: Dict[str, str]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    progress.current_query = serp_query['query']
                    if on_progress:
                        on_progress(progress)

                    from .. import GPTResearcher
                    researcher = GPTResearcher(
                        query=serp_query['query'],
                        report_type=ReportType.ResearchReport.value,
                        report_source=ReportSource.Web.value,
                        tone=self.tone,
                        websocket=self.websocket,
                        config_path=self.config_path,
                        headers=self.headers,
                        visited_urls=self.visited_urls,
                        # Propagate MCP configuration to nested researchers
                        mcp_configs=self.researcher.mcp_configs,
                        mcp_strategy=self.researcher.mcp_strategy
                    )
                    # Keep trial-level reranking anchored to the original review
                    # question, not the generated breadth/nested search query.
                    researcher.original_research_query = self.researcher.query

                    # Conduct research
                    context = await researcher.conduct_research()

                    # Get results and visited URLs
                    visited = researcher.visited_urls
                    sources = researcher.research_sources

                    # Process results to extract learnings and citations
                    results = await self.process_research_results(
                        query=serp_query['query'],
                        context=context
                    )

                    # Update progress
                    progress.completed_queries += 1
                    progress.current_breadth += 1
                    if on_progress:
                        on_progress(progress)

                    return {
                        'learnings': results['learnings'],
                        'visited_urls': list(visited),
                        'followUpQuestions': results['followUpQuestions'],
                        'researchGoal': serp_query['researchGoal'],
                        'citations': results['citations'],
                        'context': "\n".join(context) if isinstance(context, list) else (context or ""),
                        'sources': sources if sources else [],
                        'generatedQuery': serp_query['query'],
                    }

                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error processing query '{serp_query['query']}': {str(e)}")
                    print(f"\n❌ DEEP RESEARCH ERROR: {str(e)}\n{error_details}", flush=True)
                    return None
                

        # Process queries concurrently with limit
        tasks = [process_query(query) for query in serp_queries]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # Update breadth progress based on successful queries
        progress.current_breadth = len(results)
        if on_progress:
            on_progress(progress)

        # Collect all results
        for result in results:
            all_learnings.extend(result['learnings'])
            all_visited_urls.update(result['visited_urls'])
            merge_citation_maps(all_citations, result['citations'])
            if result['context']:
                all_context.append(result['context'])
            if result['sources']:
                all_sources.extend(result['sources'])

            # Continue deeper if needed
            if depth > 1:
                new_breadth = max(2, breadth // 2)
                new_depth = depth - 1
                progress.current_depth += 1

                # Create next query from research goal and follow-up questions
                # Keep recursive retrieval prompts compact and retrieval-friendly
                next_query = build_recursive_query(
                    research_goal=result["researchGoal"],
                    follow_up_questions=result["followUpQuestions"]
                )

                # Recursive research
                deeper_results = await self.deep_research(
                    query=next_query,
                    breadth=new_breadth,
                    depth=new_depth,
                    learnings=all_learnings,
                    citations=all_citations,
                    visited_urls=all_visited_urls,
                    on_progress=on_progress
                )

                all_learnings = deeper_results['learnings']
                all_visited_urls.update(deeper_results['visited_urls'])
                merge_citation_maps(all_citations, deeper_results['citations'])
                if deeper_results.get('context'):
                    all_context.extend(deeper_results['context'])
                if deeper_results.get('sources'):
                    all_sources.extend(deeper_results['sources'])
            run_record["results"].append({
                "generated_query": result["generatedQuery"],
                "research_goal": result["researchGoal"],
                "follow_up_questions": result["followUpQuestions"],
                "visited_urls": result["visited_urls"],
                "learning_count": len(result["learnings"]),
            })

        if (
            trial_level_global_dedupe_enabled()
            and os.getenv("TRIAL_LEVEL_CONTEXT", "0").strip().lower() in {"1", "true", "yes", "on"}
        ):
            before_items = len(all_context)
            before_ncts = len(set(nct for item in all_context for nct in extract_nct_ids_from_text(str(item))))
            all_context = dedupe_trial_level_context_blocks(all_context)
            after_ncts = len(set(nct for item in all_context for nct in extract_nct_ids_from_text(str(item))))
            logger.info(
                "Trial-level global dedupe changed context from %d breadth items / %d NCTs to %d trial blocks / %d NCTs",
                before_items,
                before_ncts,
                len(all_context),
                after_ncts,
            )

        # Update class tracking
        self.context.extend(all_context)
        self.research_sources.extend(all_sources)

        # Trim context to stay within word limits
        # trimmed_context = trim_context_to_word_limit(all_context)
        # logger.info(f"Trimmed context from {len(all_context)} items to {len(trimmed_context)} items to stay within word limit")

        subquery_dedupe_enabled = (
            os.getenv("SUBQUERY_NCT_DEDUPE", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        # Preserve the historical path when dedupe is disabled. Reranked
        # compression and trial-card runs share the same fair final packer.
        legacy_trim_default = "0" if subquery_dedupe_enabled else "1"
        legacy_context_trim = (
            os.getenv("LEGACY_CONTEXT_TRIM", legacy_trim_default).strip().lower()
            in {"1", "true", "yes", "on"}
        )

        if legacy_context_trim:
            trimmed_context = trim_context_to_word_limit(all_context)
            # NOTE: logger.info() alone was confirmed silently swallowed
            # throughout this file (no handler configured for this logger),
            # so a print is added here too -- this is the only reliable way
            # to actually observe which branch fired in a run's log.
            msg = f"Applied legacy intermediate context trimming: {len(all_context)} to {len(trimmed_context)} items"
            logger.info(msg)
            print(f"🪚 {msg}", flush=True)
        else:
            # A shared final budget is applied later to both compressed and
            # trial-level representations.
            trimmed_context = all_context
            msg = f"Skipped legacy intermediate context trimming: {len(all_context)} items preserved for final packing"
            logger.info(msg)
            print(f"🪚 {msg}", flush=True)

        return {
            # 'learnings': list(set(all_learnings)),
            'learnings': dedupe_strings(all_learnings),
            'visited_urls': list(all_visited_urls),
            'citations': all_citations,
            'context': trimmed_context,
            'sources': all_sources
        }

    async def run(self, on_progress=None) -> str:
        """Run the deep research process and generate final report"""
        print(f"\n🔍 DEEP RESEARCH: Starting with breadth={self.breadth}, depth={self.depth}, concurrency={self.concurrency_limit}", flush=True)
        start_time = time.time()

        # Log initial costs
        initial_costs = self.researcher.get_costs()

        # ------------ OG ----------------------------------

        # follow_up_questions = await self.generate_research_plan(self.researcher.query)
        # answers = ["Automatically proceeding with research"] * len(follow_up_questions)

        # qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)]
        # combined_query = f"""
        # Initial Query: {self.researcher.query}\nFollow - up Questions and Answers:\n
        # """ + "\n".join(qa_pairs)

        # ---------------------------------------------------

        # -------------- V1 ---------------------------------
        # combined_query = self.researcher.query
        # -----------------------------------------------------

        # ------------- V2 -----------------------------------
        # search_directions = await self.generate_research_plan(self.researcher.query)

        # combined_query = (
        #     f"Original research question:\n{self.researcher.query}\n\n"
        #     "Additional registry search directions:\n"
        #     + "\n".join(f"- {q}" for q in search_directions)
        #     + "\n\nUse these only as retrieval directions. Do not treat them as answered evidence."
        # )
        # ------------------------------------------------------------------

        # --------------- V3 ------------------------------------
        search_directions = await self.generate_research_plan(self.researcher.query)
        search_directions = dedupe_strings(search_directions, max_items=self.breadth)

        combined_query = build_combined_query(
            original_query=self.researcher.query,
            search_directions=search_directions
        )

        self.query_trace["combined_query"] = combined_query
        self.query_trace["breadth"] = self.breadth
        self.query_trace["depth"] = self.depth
                # --------------------------------------------------------
        results = await self.deep_research(
            query=combined_query,
            breadth=self.breadth,
            depth=self.depth,
            on_progress=on_progress
        )

        # Get costs after deep research
        research_costs = self.researcher.get_costs() - initial_costs

        # Log research costs if we have a log handler
        if self.researcher.log_handler:
            await self.researcher._log_event("research", step="deep_research_costs", details={
                "research_costs": research_costs,
                "total_costs": self.researcher.get_costs()
            })

        # Prepare context with citations
        # context_with_citations = []
        # for learning in results['learnings']:
        #     citation = results['citations'].get(learning, '')
        #     if citation:
        #         context_with_citations.append(f"{learning} [Source: {citation}]")
        #     else:
        #         context_with_citations.append(learning)

        # # Add all research context
        # if results.get('context'):
        #     context_with_citations.extend(results['context'])

        # # Trim final context to word limit
        # final_context = trim_context_to_word_limit(context_with_citations)

        # Prepare concise learning context with citations.
        learning_context = []

        for learning in results["learnings"]:
            citation = results["citations"].get(learning, "")

            if citation:
                learning_context.append(f"{learning} [Source: {citation}]")
            else:
                learning_context.append(learning)

        # Preserve the original trial-level evidence separately.
        trial_context = list(results.get("context") or [])

        if trial_level_learning_dedupe_enabled():
            trial_context = select_trial_context_fallback(
                trial_context,
                learning_context,
            )

        max_context_words = int(
            os.getenv("FINAL_CONTEXT_MAX_WORDS", str(MAX_CONTEXT_WORDS))
        )
        learning_word_budget = int(
            os.getenv("FINAL_CONTEXT_LEARNING_WORDS", "3000")
        )

        learning_word_budget = min(
            max(0, learning_word_budget),
            max_context_words,
        )

        trimmed_learnings = trim_context_from_start(
            learning_context,
            max_words=learning_word_budget,
        )

        # Unused learning capacity is returned to the trial-card budget.
        learning_words_used = count_words(trimmed_learnings)
        trial_word_budget = max(0, max_context_words - learning_words_used)

        # trimmed_trials = trim_context_to_word_limit(
        #     trial_context,
        #     max_words=trial_word_budget,
        # )
        ordered_trial_blocks = interleave_trial_context_blocks(trial_context)

        trimmed_trials = pack_context_from_start(
            ordered_trial_blocks,
            max_words=trial_word_budget,
        )

        final_context = trimmed_learnings + trimmed_trials

        # Keep the full pre-trim representation for debugging.
        context_with_citations = learning_context + trial_context

        logger.info(
            "Final context budget: %d/%d learning items (%d words), "
            "%d/%d trial-context items (%d-word budget)",
            len(trimmed_learnings),
            len(learning_context),
            learning_words_used,
            len(trimmed_trials),
            # len(trial_context),
            len(ordered_trial_blocks), 
            trial_word_budget,
        )
        self._save_final_context_debug(context_with_citations, final_context, results)
        
        # Set enhanced context and visited URLs
        self.researcher.context = "\n".join(final_context)
        self.researcher.visited_urls = results['visited_urls']

        # Set research sources
        if results.get('sources'):
            self.researcher.research_sources = results['sources']

        # Log total execution time
        end_time = time.time()
        execution_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Total research execution time: {execution_time}")
        logger.info(f"Total research costs: ${research_costs:.2f}")

        # Return the context - don't generate report here as it will be done by the main agent
        return self.researcher.context
