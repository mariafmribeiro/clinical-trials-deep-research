"""Research conductor skill for GPT Researcher.

This module provides the ResearchConductor class that manages and
coordinates the research process including query planning, web searching,
and context gathering.
"""

import asyncio
import logging
import os
import random
import re

from ..actions.agent_creator import choose_agent
from ..actions.query_processing import get_search_results, plan_research_outline
from ..actions.utils import stream_output
from ..document import DocumentLoader, LangChainDocumentLoader, OnlineDocumentLoader
from ..utils.enum import ReportSource, ReportType
from ..utils.logging_config import get_json_handler
import json
from datetime import datetime
from pathlib import Path

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion

NCT_ID_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


class ResearchConductor:
    """Manages and coordinates the research process.

    This class handles the main research workflow including planning
    research queries, conducting web searches, managing MCP retrievers,
    and gathering context from various sources.

    Attributes:
        researcher: The parent GPTResearcher instance.
        logger: Logger for research events.
        json_handler: Handler for JSON logging.
    """

    def __init__(self, researcher):
        """Initialize the ResearchConductor.

        Args:
            researcher: The GPTResearcher instance that owns this conductor.
        """
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()
        # Add cache for MCP results to avoid redundant calls
        self._mcp_results_cache = None
        # Track MCP query count for balanced mode
        self._mcp_query_count = 0

# ------- PARA FAZER TRACK DO CONTEXTO --------
    def _save_web_context_debug(self, sub_query: str, scraped_data: list, web_context: str) -> None:
        if os.getenv("SAVE_WEB_CONTEXT_DEBUG", "0") != "1":
            return

        base_dir = Path(
            os.getenv(
                "WEB_CONTEXT_DEBUG_DIR",
                os.getenv("FINAL_CONTEXT_DEBUG_DIR", str(Path("logs") / "final_context_debug")),
            )
        )
        output_dir = base_dir / "web_context"
        output_dir.mkdir(parents=True, exist_ok=True)

        label = os.getenv("FINAL_CONTEXT_DEBUG_LABEL", "research")
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "research"
        safe_query = re.sub(r"[^A-Za-z0-9._-]+", "_", sub_query[:80]).strip("_") or "query"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        stem = f"{safe_label}_{stamp}_{safe_query}"
        web_context_text = str(web_context or "")

        nct_ids = sorted({nct.upper() for nct in NCT_ID_RE.findall(web_context_text)})
        scraped_nct_ids = sorted({
            str(page.get("nct_id", "")).upper()
            for page in scraped_data or []
            if page.get("nct_id")
        })

        (output_dir / f"{stem}_web_context.txt").write_text(
            web_context_text,
            encoding="utf-8",
        )

        (output_dir / f"{stem}_web_context_meta.json").write_text(
            json.dumps(
                {
                    "sub_query": sub_query,
                    "web_context_chars": len(web_context_text),
                    "web_context_nct_count": len(nct_ids),
                    "web_context_nct_ids": nct_ids,
                    "scraped_candidate_count": len(scraped_data or []),
                    "scraped_candidate_nct_count": len(scraped_nct_ids),
                    "scraped_candidate_nct_ids": scraped_nct_ids,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

# -------------------------------------------------------------------------------------------

    async def plan_research(self, query, query_domains=None):
        """Gets the sub-queries from the query
        Args:
            query: original query
        Returns:
            List of queries
        """
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        search_results = await get_search_results(query, self.researcher.retrievers[0], query_domains, researcher=self.researcher)
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")

        # This search_results list gets dumped straight into the
        # sub-query-writing prompt as context. Cap each result's text so
        # query generation reacts to registry/PICO-level details, not
        # posted clinical-results text — which the trial card now puts
        # last, so this cap naturally drops it first. The uncapped card is
        # still used downstream (reranker, final trial-level context).
        # Also cap how MANY results go in: up to 300 can come back from a
        # single search, and with no cap here that's up to 300 x 2000 chars
        # in one prompt, well past what the model's context window can hold,
        # risking silent truncation of the task/instructions themselves.
        # Results are already BM25-ranked, so keeping the first N is keeping
        # the most relevant ones, not an arbitrary subset.
        plan_context_results = self._int_env("PLAN_RESEARCH_CONTEXT_RESULTS", 15, 1)
        search_results = search_results[:plan_context_results]
        plan_context_chars = self._int_env("PLAN_RESEARCH_CONTEXT_CHARS", 2000, 200)
        search_results = [
            {
                **result,
                **{
                    key: str(result[key])[:plan_context_chars]
                    for key in ("raw_content", "body", "content")
                    if result.get(key)
                },
            }
            if isinstance(result, dict)
            else result
            for result in search_results
        ]

        await stream_output(
            "logs",
            "planning_research",
            f"🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        # Remove duplicate logging - this will be logged once in conduct_research instead

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
            retriever_names=retriever_names,  # Pass retriever names for MCP optimization
            **self.researcher.kwargs
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """Runs the GPT Researcher to conduct research"""
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        
        # Log active retrievers once at the start of research
        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        self.logger.info(f"Active retrievers: {retriever_names}")
        
        # Reset visited_urls and source_urls at the start of each research task
        self.researcher.visited_urls.clear()
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "agent_generated",
                self.researcher.agent,
                self.researcher.websocket
            )

        # Choose agent and role if not already defined
        if not (self.researcher.agent and self.researcher.role):
            self.researcher.agent, self.researcher.role = await choose_agent(
                query=self.researcher.query,
                cfg=self.researcher.cfg,
                parent_query=self.researcher.parent_query,
                cost_callback=self.researcher.add_costs,
                headers=self.researcher.headers,
                prompt_family=self.researcher.prompt_family
            )
                
        # Check if MCP retrievers are configured
        has_mcp_retriever = any("mcpretriever" in r.__name__.lower() for r in self.researcher.retrievers)
        if has_mcp_retriever:
            self.logger.info("MCP retrievers configured and will be used with standard research flow")

        # Conduct research based on the source type
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            if research_data and len(research_data) == 0 and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    f"🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data += ' '.join(additional_research)
        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search with all configured retrievers")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
        elif self.researcher.report_source == ReportSource.Local.value:
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)

            research_data = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
        # Hybrid search including both local documents and web sources
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            if self.researcher.document_urls:
                document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
            else:
                document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            docs_context = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
            web_context = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)
        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # Reuse existing loader
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)
            
        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )
        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # Rank and curate the sources
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            self.researcher.context = await self.researcher.source_curator.curate_sources(research_data)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """Scrapes and compresses the context from the given urls"""
        self.logger.info(f"Getting context from URLs: {urls}")
        
        new_search_urls = await self._get_new_urls(urls)
        self.logger.info(f"New URLs to process: {new_search_urls}")

        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        context = await self.researcher.context_manager.get_similar_content_by_query(
            self.researcher.query, scraped_content
        )
        return context

    # Add logging to other methods similarly...

    async def _get_context_by_vectorstore(self, query, filter: dict | None = None):
        """
        Generates the context for the research task by searching the vectorstore
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query)
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context

    async def _get_context_by_web_search(self, query, scraped_data: list | None = None, query_domains: list | None = None):
        """
        Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []

        # **CONFIGURABLE MCP OPTIMIZATION: Control MCP strategy**
        mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
        
        # Get MCP strategy configuration
        mcp_strategy = self._get_mcp_strategy()
        
        if mcp_retrievers and self._mcp_results_cache is None:
            if mcp_strategy == "disabled":
                # MCP disabled - skip MCP research entirely
                self.logger.info("MCP disabled by strategy, skipping MCP research")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_disabled",
                        f"⚡ MCP research disabled by configuration",
                        self.researcher.websocket,
                    )
            elif mcp_strategy == "fast":
                # Fast: Run MCP once with original query
                self.logger.info("MCP fast strategy: Running once with original query")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_optimization",
                        f"🚀 MCP Fast: Running once for main query (performance mode)",
                        self.researcher.websocket,
                    )
                
                # Execute MCP research once with the original query
                mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                self._mcp_results_cache = mcp_context
                self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")
            elif mcp_strategy == "deep":
                # Deep: Will run MCP for all queries (original behavior) - defer to per-query execution
                self.logger.info("MCP deep strategy: Will run for all queries")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_comprehensive",
                        f"🔍 MCP Deep: Will run for each sub-query (thorough mode)",
                        self.researcher.websocket,
                    )
                # Don't cache - let each sub-query run MCP individually
            else:
                # Unknown strategy - default to fast
                self.logger.warning(f"Unknown MCP strategy '{mcp_strategy}', defaulting to fast")
                mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                self._mcp_results_cache = mcp_context
                self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")

        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query, query_domains)
        self.logger.info(f"Generated sub-queries: {sub_queries}")
        
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        if (
            self._subquery_nct_dedupe_enabled()
            and not scraped_data
            and not mcp_retrievers
        ):
            return await self._get_context_by_subquery_nct_dedupe(
                query=query,
                sub_queries=sub_queries,
                query_domains=query_domains,
            )

        if self._subquery_nct_dedupe_enabled() and (scraped_data or mcp_retrievers):
            self.logger.info(
                "SUBQUERY_NCT_DEDUPE requested but skipped because scraped_data or MCP retrievers are active"
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        try:
            context = await asyncio.gather(
                *[
                    self._process_sub_query(sub_query, scraped_data, query_domains)
                    for sub_query in sub_queries
                ]
            )
            self.logger.info(f"Gathered context from {len(context)} sub-queries")
            # Filter out empty results and join the context
            context = [c for c in context if c]
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"Combined context size: {len(combined_context)}")
                return combined_context
            return []
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return []

    def _get_mcp_strategy(self) -> str:
        """
        Get the MCP strategy configuration.
        
        Priority:
        1. Instance-level setting (self.researcher.mcp_strategy)
        2. Config file setting (self.researcher.cfg.mcp_strategy) 
        3. Default value ("fast")
        
        Returns:
            str: MCP strategy
                "disabled" = Skip MCP entirely
                "fast" = Run MCP once with original query (default)
                "deep" = Run MCP for all sub-queries
        """
        # Check instance-level setting first
        if hasattr(self.researcher, 'mcp_strategy') and self.researcher.mcp_strategy is not None:
            return self.researcher.mcp_strategy
        
        # Check config setting
        if hasattr(self.researcher.cfg, 'mcp_strategy'):
            return self.researcher.cfg.mcp_strategy
        
        # Default to fast mode
        return "fast"

    def _subquery_nct_dedupe_enabled(self) -> bool:
        value = os.getenv("SUBQUERY_NCT_DEDUPE", "0").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _extract_nct_id_from_page(self, page: dict) -> str | None:
        for key in ("nct_id", "nctId", "id", "trial_id"):
            value = page.get(key)
            if value:
                match = NCT_ID_RE.search(str(value))
                if match:
                    return match.group(0).upper()

        text = " ".join(
            str(page.get(key, ""))
            for key in ("url", "href", "title", "body", "raw_content")
        )
        match = NCT_ID_RE.search(text)
        return match.group(0).upper() if match else None

    def _score_from_page(self, page: dict) -> float:
        for key in ("score", "_score", "search_score"):
            try:
                return float(page.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _retrieval_rrf_score(self, ranks) -> float:
        """Score by the single best per-query retrieval rank, not summed RRF.

        Classic RRF sums 1/(rrf_k+rank) across every query a trial appeared
        in, which rewards trials that show up in many queries at mediocre
        rank over trials that rank excellently in just one narrow query.
        Ground-truth trials here are often findable only via one specific
        query (e.g. a boolean query naming an uncommon drug), so instead we
        score only the best (lowest) rank achieved in any single query.
        Keeps the same env var / smoothing constant and the same "higher is
        better" convention as before, so callers don't need to change.
        """
        rrf_k = max(1, int(os.getenv("LLM_NCT_RERANK_RRF_K", "60")))
        valid_ranks = []
        for rank in ranks or []:
            try:
                valid_ranks.append(max(1, int(rank)))
            except (TypeError, ValueError):
                continue
        if not valid_ranks:
            return 0.0
        return 1.0 / (rrf_k + min(valid_ranks))

    def _llm_nct_rerank_enabled(self) -> bool:
        value = os.getenv("LLM_NCT_RERANK", "0").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _trial_level_context_enabled(self) -> bool:
        value = os.getenv("TRIAL_LEVEL_CONTEXT", "0").strip().lower()
        return value in {"1", "true", "yes", "on"}  

    def _int_env(self, name: str, default: int, minimum: int | None = None) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _truncate_for_rerank(self, value, max_chars: int) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value if item is not None)
        value = re.sub(r"\s+", " ", str(value)).strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rsplit(" ", 1)[0].rstrip() + " ..."

    def _extract_card_field(self, raw_content: str, label: str, max_chars: int) -> str:
        if not raw_content:
            return ""
        pattern = re.compile(rf"^{re.escape(label)}:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(raw_content)
        return self._truncate_for_rerank(match.group(1), max_chars) if match else ""

    # def _compact_trial_card_for_llm_rerank(self, page: dict) -> str:
    #     raw_content = str(page.get("raw_content") or "")
    #     nct_id = page.get("nct_id") or self._extract_nct_id_from_page(page) or "NO_NCT_ID"
    #     title = self._truncate_for_rerank(
    #         page.get("title") or self._extract_card_field(raw_content, "Brief title", 160),
    #         180,
    #     )
    #     conditions = self._extract_card_field(raw_content, "Conditions", 180)
    #     interventions = self._extract_card_field(raw_content, "Interventions", 220)
    #     study_type = self._extract_card_field(raw_content, "Study type", 80)
    #     outcomes = self._extract_card_field(raw_content, "Primary outcomes", 260)
    #     summary = self._extract_card_field(raw_content, "Brief summary", 360)
    #     hit_count = page.get("retrieval_hit_count", 1)
    #     best_rank = page.get("best_retrieval_rank", page.get("retrieval_rank", ""))

    #     parts = [
    #         f"NCT ID: {nct_id}",
    #         f"Title: {title}" if title else "",
    #         f"Conditions: {conditions}" if conditions else "",
    #         f"Interventions: {interventions}" if interventions else "",
    #         f"Study type: {study_type}" if study_type else "",
    #         f"Primary outcomes: {outcomes}" if outcomes else "",
    #         f"Brief summary: {summary}" if summary else "",
    #         f"Retrieval evidence: hit_count={hit_count}; best_rank={best_rank}",
    #     ]
    #     return "\n".join(part for part in parts if part)
    # def _compact_trial_card_for_llm_rerank(self, page: dict) -> str:
    #         raw_content = str(page.get("raw_content") or "")
    #         nct_id = page.get("nct_id") or self._extract_nct_id_from_page(page) or "NO_NCT_ID"
    #         title = self._truncate_for_rerank(
    #             page.get("title") or self._extract_card_field(raw_content, "Brief title", 160),
    #             180,
    #         )
    #         conditions = self._extract_card_field(raw_content, "Conditions", 180)
    #         interventions = self._extract_card_field(raw_content, "Interventions", 240)
    #         arms = self._extract_card_field(raw_content, "Arms / groups", 220)
    #         study_type = self._extract_card_field(raw_content, "Study type", 100)
    #         study_design = self._extract_card_field(raw_content, "Study design", 220)
    #         outcomes = self._extract_card_field(raw_content, "Primary outcomes", 240)
    #         eligibility = self._extract_card_field(raw_content, "Eligibility / population", 320)
    #         summary = self._extract_card_field(raw_content, "Brief summary", 320)
    #         hit_count = page.get("retrieval_hit_count", 1)
    #         best_rank = page.get("best_retrieval_rank", page.get("retrieval_rank", ""))
    
    #         parts = [
    #             f"NCT ID: {nct_id}",
    #             f"Title: {title}" if title else "",
    #             f"Conditions: {conditions}" if conditions else "",
    #             f"Interventions: {interventions}" if interventions else "",
    #             f"Arms/groups: {arms}" if arms else "",
    #             f"Study type: {study_type}" if study_type else "",
    #             f"Study design: {study_design}" if study_design else "",
    #             f"Primary outcomes: {outcomes}" if outcomes else "",
    #             f"Eligibility/population: {eligibility}" if eligibility else "",
    #             f"Brief summary: {summary}" if summary else "",
    #             f"Retrieval evidence: hit_count={hit_count}; best_rank={best_rank}",
    #         ]
    #         return "\n".join(part for part in parts if part)
    def _compact_trial_card_for_llm_rerank(self, page: dict) -> str:
        raw_content = str(page.get("raw_content") or "")
        nct_id = page.get("nct_id") or self._extract_nct_id_from_page(page) or "NO_NCT_ID"
        title = self._truncate_for_rerank(
            page.get("title") or self._extract_card_field(raw_content, "Brief title", 160),
            220,
        )
        interventions = self._extract_card_field(raw_content, "Interventions", 300)
        summary = self._extract_card_field(raw_content, "Brief summary", 1200)
        hit_count = page.get("retrieval_hit_count", 1)
        best_rank = page.get("best_retrieval_rank", page.get("retrieval_rank", ""))

        parts = [
            f"NCT ID: {nct_id}",
            f"Title: {title}" if title else "",
            f"Interventions: {interventions}" if interventions else "",
            f"Brief summary: {summary}" if summary else "",
            f"Retrieval evidence: hit_count={hit_count}; best_rank={best_rank}",
        ]
        return "\n".join(part for part in parts if part)

    def _compact_trial_paragraph_for_context(self, page: dict, index: int, max_chars: int) -> str:
        raw_content = str(page.get("raw_content") or "")
        nct_id = page.get("nct_id") or self._extract_nct_id_from_page(page) or "NO_NCT_ID"
        source = page.get("url") or page.get("href") or f"local://{nct_id}"

        title = self._truncate_for_rerank(
            page.get("title") or self._extract_card_field(raw_content, "Brief title", 220),
            self._int_env("TRIAL_LEVEL_CONTEXT_TITLE_CHARS", 220, 40),
        )
        conditions = self._extract_card_field(
            raw_content,
            "Conditions",
            self._int_env("TRIAL_LEVEL_CONTEXT_CONDITIONS_CHARS", 220, 40),
        )
        interventions = self._extract_card_field(
            raw_content,
            "Interventions",
            self._int_env("TRIAL_LEVEL_CONTEXT_INTERVENTIONS_CHARS", 280, 60),
        )
        study_type = self._extract_card_field(raw_content, "Study type", 120)
        design = self._extract_card_field(
            raw_content,
            "Study design",
            self._int_env("TRIAL_LEVEL_CONTEXT_DESIGN_CHARS", 220, 40),
        )
        primary_outcomes = self._extract_card_field(
            raw_content,
            "Primary outcomes",
            self._int_env("TRIAL_LEVEL_CONTEXT_OUTCOMES_CHARS", 320, 80),
        )
        secondary_outcomes = self._extract_card_field(
            raw_content,
            "Secondary outcomes",
            self._int_env("TRIAL_LEVEL_CONTEXT_SECONDARY_OUTCOMES_CHARS", 220, 60),
        )
        summary = self._extract_card_field(
            raw_content,
            "Brief summary",
            self._int_env("TRIAL_LEVEL_CONTEXT_SUMMARY_CHARS", 420, 120),
        )
        outcome_results = self._extract_card_field(
            raw_content,
            "Clinical outcome results",
            self._int_env("TRIAL_LEVEL_CONTEXT_OUTCOME_RESULTS_CHARS", 500, 0),
        )
        safety_results = self._extract_card_field(
            raw_content,
            "Clinical safety results",
            self._int_env("TRIAL_LEVEL_CONTEXT_SAFETY_RESULTS_CHARS", 220, 0),
        )
        eligibility = self._extract_card_field(
            raw_content,
            "Eligibility / population",
            self._int_env("TRIAL_LEVEL_CONTEXT_ELIGIBILITY_CHARS", 260, 0),
        )

        # Ranking metadata has already served its purpose during selection and
        # is not clinical evidence. Put posted values before descriptive fields
        # so the total card budget cannot silently remove them.
        lines = [
            f"### Trial {index}: {nct_id}",
            f"Source: {source}",
            f"Title: {title}" if title else "",
            f"Intervention/comparator: {interventions}" if interventions else "",
            f"Posted outcome results: {outcome_results}" if outcome_results else "",
            f"Posted safety results: {safety_results}" if safety_results else "",
            f"Population/condition: {conditions}" if conditions else "",
            f"Primary outcomes: {primary_outcomes}" if primary_outcomes else "",
            f"Secondary outcomes: {secondary_outcomes}" if secondary_outcomes else "",
            f"Study type/design: {'; '.join(part for part in [study_type, design] if part)}" if (study_type or design) else "",
            f"Brief summary: {summary}" if summary else "",
            f"Eligibility/population details: {eligibility}" if eligibility else "",
        ]

        kept_lines = []
        used_chars = 0
        for line in (line for line in lines if line):
            separator_chars = 1 if kept_lines else 0
            remaining = max_chars - used_chars - separator_chars
            if remaining <= 0:
                break

            if len(line) > remaining:
                if remaining < 40:
                    break
                line = self._truncate_for_rerank(line, remaining)
                line = line[:remaining]

            kept_lines.append(line)
            used_chars += separator_chars + len(line)

        return "\n".join(kept_lines)

    def _format_trial_level_context(self, query: str, candidates: list[dict]) -> str:
        top_k = self._int_env("TRIAL_LEVEL_CONTEXT_TOP_K", 30, 1)
        max_chars = self._int_env("TRIAL_LEVEL_CONTEXT_MAX_CHARS_PER_TRIAL", 1600, 300)
        min_llm_score = self._int_env("TRIAL_LEVEL_CONTEXT_MIN_LLM_SCORE", -1)
        fallback_k = self._int_env("TRIAL_LEVEL_CONTEXT_FALLBACK_K", 0, 0)
        fallback_max_rank = self._int_env("TRIAL_LEVEL_CONTEXT_FALLBACK_MAX_RANK", top_k, top_k)
        fallback_min_score = self._int_env("TRIAL_LEVEL_CONTEXT_FALLBACK_MIN_LLM_SCORE", 2)

        keyed_candidates = [
            candidate for candidate in candidates
            if isinstance(candidate, dict) and (candidate.get("nct_id") or self._extract_nct_id_from_page(candidate))
        ]

        if min_llm_score >= 0:
            keyed_candidates = [
                candidate for candidate in keyed_candidates
                if int(candidate.get("llm_rerank_score", -1)) >= min_llm_score
            ]

        selected_candidates = list(keyed_candidates[:top_k])

        if fallback_k > 0 and fallback_max_rank > top_k:
            selected_ncts = {
                candidate.get("nct_id") or self._extract_nct_id_from_page(candidate)
                for candidate in selected_candidates
            }
            selected_ncts = {nct_id for nct_id in selected_ncts if nct_id}
            fallback_candidates = []

            for candidate in keyed_candidates[top_k:fallback_max_rank]:
                nct_id = candidate.get("nct_id") or self._extract_nct_id_from_page(candidate)
                if not nct_id or nct_id in selected_ncts:
                    continue
                try:
                    score = int(candidate.get("llm_rerank_score", -1))
                except (TypeError, ValueError):
                    score = -1
                if score < fallback_min_score:
                    continue
                fallback_candidates.append(candidate)
                selected_ncts.add(nct_id)
                if len(fallback_candidates) >= fallback_k:
                    break

            selected_candidates.extend(fallback_candidates)

        if not selected_candidates:
            return ""

        trial_blocks = [
            self._compact_trial_paragraph_for_context(candidate, idx, max_chars)
            for idx, candidate in enumerate(selected_candidates, start=1)
        ]
        return "\n\n".join(trial_blocks)

    def _load_llm_json_payload(self, response: str):
        text = (response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        candidates = [text]
        for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
            match = re.search(pattern, text)
            if match:
                candidates.append(match.group(0))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _parse_llm_nct_rerank_response(self, response: str) -> tuple[list[str], dict[str, str]]:
        text = (response or "").strip()
        selected: list[str] = []
        labels: dict[str, str] = {}

        parsed = self._load_llm_json_payload(text)
        if parsed is None:
            for match in re.finditer(r"\bNCT\d{8}\b", text, flags=re.IGNORECASE):
                nct_id = match.group(0).upper()
                if nct_id not in selected:
                    selected.append(nct_id)
                    labels[nct_id] = "MAYBE"
            return selected, labels

        items = parsed.get("trials") if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            items = []

        for item in items:
            if isinstance(item, str):
                nct_match = NCT_ID_RE.search(item)
                label = "MAYBE"
            elif isinstance(item, dict):
                nct_match = NCT_ID_RE.search(str(item.get("nct_id") or item.get("id") or ""))
                label = str(item.get("label") or item.get("classification") or "MAYBE").upper()
            else:
                continue

            if not nct_match:
                continue

            nct_id = nct_match.group(0).upper()
            if label not in {"INCLUDE", "MAYBE", "EXCLUDE"}:
                label = "MAYBE"
            labels[nct_id] = label
            if label in {"INCLUDE", "MAYBE"} and nct_id not in selected:
                selected.append(nct_id)

        return selected, labels

    def _parse_llm_nct_score_response(self, response: str) -> dict[str, dict]:
        text = response or ""
        parsed = self._load_llm_json_payload(text)
        scored: dict[str, dict] = {}

        if parsed is None:
            # Last-resort fallback: preserve mentioned order with a neutral score.
            for order, match in enumerate(re.finditer(r"\bNCT\d{8}\b", text, flags=re.IGNORECASE)):
                nct_id = match.group(0).upper()
                scored.setdefault(nct_id, {"score": 2, "order": order})
            return scored

        items = parsed.get("trials") if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            return scored

        for order, item in enumerate(items):
            if isinstance(item, str):
                nct_match = NCT_ID_RE.search(item)
                score = 2
            elif isinstance(item, dict):
                nct_match = NCT_ID_RE.search(str(item.get("nct_id") or item.get("id") or ""))
                raw_score = item.get("score", item.get("eligibility_score", item.get("rank_score", 2)))
                try:
                    score = int(float(raw_score))
                except (TypeError, ValueError):
                    score = 2
            else:
                continue

            if not nct_match:
                continue

            nct_id = nct_match.group(0).upper()
            score = max(0, min(3, score))
            scored[nct_id] = {"score": score, "order": order}

        return scored

    def _save_llm_nct_rerank_debug(self, payload: dict) -> None:
        if os.getenv("SAVE_LLM_NCT_RERANK_DEBUG", os.getenv("SAVE_WEB_CONTEXT_DEBUG", "0")) != "1":
            return

        base_dir = Path(
            os.getenv(
                "LLM_NCT_RERANK_DEBUG_DIR",
                os.getenv("FINAL_CONTEXT_DEBUG_DIR", str(Path("logs") / "final_context_debug")),
            )
        )
        output_dir = base_dir / "llm_nct_rerank"
        output_dir.mkdir(parents=True, exist_ok=True)

        label = os.getenv("FINAL_CONTEXT_DEBUG_LABEL", "research")
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "research"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = output_dir / f"{safe_label}_{stamp}_llm_nct_rerank.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _select_stratified_rerank_candidates(
        self,
        candidates: list[dict],
        input_limit: int,
    ) -> list[dict]:
        """Build a balanced reranker pool from the ranked subquery results."""
        if input_limit <= 0 or not candidates:
            return []
        if len(candidates) <= input_limit:
            return list(candidates)

        query_order: list[str] = []
        seen_queries = set()
        for candidate in candidates:
            query_names = list(candidate.get("retrieved_by_queries") or [])
            query_names.extend(
                (candidate.get("retrieval_ranks_by_query") or {}).keys()
            )
            for query_name in query_names:
                query_name = str(query_name or "").strip()
                if not query_name or query_name.startswith("__unknown_"):
                    continue
                if query_name not in seen_queries:
                    seen_queries.add(query_name)
                    query_order.append(query_name)

        def retrieval_key(candidate: dict, query_name: str | None = None):
            if query_name:
                rank = (candidate.get("retrieval_ranks_by_query") or {}).get(
                    query_name,
                    999999,
                )
            else:
                rank = candidate.get("best_retrieval_rank", 999999)
            try:
                rank = int(rank)
            except (TypeError, ValueError):
                rank = 999999
            try:
                hit_count = int(candidate.get("retrieval_hit_count") or 0)
            except (TypeError, ValueError):
                hit_count = 0
            try:
                search_score = float(candidate.get("best_search_score") or 0.0)
            except (TypeError, ValueError):
                search_score = 0.0
            return (
                rank,
                -hit_count,
                -search_score,
                str(candidate.get("nct_id") or ""),
            )

        per_query = {
            query_name: sorted(
                [
                    candidate
                    for candidate in candidates
                    if query_name
                    in (candidate.get("retrieval_ranks_by_query") or {})
                ],
                key=lambda candidate, query_name=query_name: retrieval_key(
                    candidate,
                    query_name,
                ),
            )
            for query_name in query_order
        }

        selected: list[dict] = []
        selected_ncts = set()
        cursors = {query_name: 0 for query_name in query_order}

        while len(selected) < input_limit and query_order:
            made_progress = False
            for query_name in query_order:
                query_candidates = per_query[query_name]
                cursor = cursors[query_name]
                while cursor < len(query_candidates):
                    candidate = query_candidates[cursor]
                    cursor += 1
                    nct_id = candidate.get("nct_id")
                    if nct_id in selected_ncts:
                        continue
                    selected.append(candidate)
                    selected_ncts.add(nct_id)
                    made_progress = True
                    break
                cursors[query_name] = cursor
                if len(selected) >= input_limit:
                    break
            if not made_progress:
                break

        # Fill any unused capacity by retrieval rank across the complete pool.
        for candidate in sorted(candidates, key=retrieval_key):
            if len(selected) >= input_limit:
                break
            nct_id = candidate.get("nct_id")
            if nct_id in selected_ncts:
                continue
            selected.append(candidate)
            selected_ncts.add(nct_id)

        self.logger.info(
            "Built stratified reranker input with %d candidates across %d subqueries",
            len(selected),
            len(query_order),
        )
        return selected

    async def _llm_rerank_nct_candidates(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        if not self._llm_nct_rerank_enabled():
            return candidates

        keyed_candidates = [item for item in candidates if item.get("nct_id")]
        unkeyed_candidates = [item for item in candidates if not item.get("nct_id")]
        if not keyed_candidates:
            return candidates

        input_limit = int(os.getenv("LLM_NCT_RERANK_INPUT_LIMIT", "120"))
        keep_limit = int(os.getenv("LLM_NCT_RERANK_KEEP_LIMIT", "100"))
        min_keep = int(os.getenv("LLM_NCT_RERANK_MIN_KEEP", "50"))
        total_limit = int(os.getenv("LLM_NCT_RERANK_TOTAL_LIMIT", "0") or 0)
        batch_size = max(1, int(os.getenv("LLM_NCT_RERANK_BATCH_SIZE", "40")))
        rrf_k = max(1, int(os.getenv("LLM_NCT_RERANK_RRF_K", "60")))
        rerank_mode = os.getenv("LLM_NCT_RERANK_MODE", "boost").strip().lower()
        max_tokens = int(os.getenv("LLM_NCT_RERANK_MAX_TOKENS", "3000"))
        rerank_query = (
            os.getenv("LLM_NCT_RERANK_QUERY", "").strip()
            or str(getattr(self.researcher, "original_research_query", "") or "").strip()
            or query
        )

        llm_candidates = self._select_stratified_rerank_candidates(
            keyed_candidates,
            input_limit,
        )
        raw_responses: list[str] = []
        labels: dict[str, str] = {}
        score_info: dict[str, dict] = {}
        selected_ncts: list[str] = []

        try:
            if rerank_mode == "rank_filter":
                for batch_start in range(0, len(llm_candidates), batch_size):
                    batch = llm_candidates[batch_start:batch_start + batch_size]
                    cards = [
                        f"[{batch_start + idx}]\n{self._compact_trial_card_for_llm_rerank(candidate)}"
                        for idx, candidate in enumerate(batch, start=1)
                    ]
                    prompt = (
                        "You are ranking ClinicalTrials.gov records for a Cochrane-style systematic review.\n"
                        "Score EACH candidate trial against the review question. Use eligibility signals: "
                        "population/condition, intervention, comparator, study design, and outcomes.\n"
                        "Be recall-oriented. If a trial might be included, give it 2 rather than 0 or 1.\n"
                        "Score meanings: 3=likely included, 2=possibly included, "
                        "1=related but probably not included, 0=not relevant.\n"
                        "Do not require clinical results to be present; registry design information can still be relevant.\n"
                        f"Review question:\n{rerank_query}\n\n"
                        "Return JSON only. Do not include markdown, prose, summaries, or explanations outside JSON.\n"
                        "Use exactly this schema:\n"
                        '{"trials":[{"nct_id":"NCT########","score":0}]}\n\n'
                        "Candidate trials:\n"
                        + "\n\n---\n\n".join(cards)
                    )
                    # prompt = (
                    #     "You are scoring ClinicalTrials.gov records for likely inclusion in a Cochrane-style systematic review.\n"
                    #     "Score EACH candidate trial against the review question using PICO eligibility: "
                    #     "population/condition, intervention, comparator/control, and study design.\n"
                    #     "Scores: "
                    #     "3=likely eligible; population and intervention match, and comparator/design are plausible. "
                    #     "Keep in context even if outcomes/results are missing. "
                    #     "2=possible but uncertain; related trial, but an important PICO element is unclear or only partly matched. "
                    #     "1=related but probably excluded; shares terms, but likely wrong population, intervention, comparator, or design. "
                    #     "0=not relevant.\n"
                    #     "Be recall-oriented: if a clinical trial plausibly matches the review and losing it would risk missing an eligible study, score 3 rather than 2.\n"
                    #     "Do not give 3 for keyword overlap alone.\n"
                    #     f"Review question:\n{rerank_query}\n\n"
                    #     "Return JSON only. Do not include markdown, prose, summaries, or explanations outside JSON.\n"
                    #     "Use exactly this schema:\n"
                    #     '{"trials":[{"nct_id":"NCT########","score":0,"label":"INCLUDE|MAYBE|EXCLUDE"}]}\n\n'
                    #     "Candidate trials:\n"
                    #     + "\n\n---\n\n".join(cards)
                    # )
                    # prompt = (
                    #     "You are scoring ClinicalTrials.gov records for likely inclusion in a Cochrane-style systematic review.\n"
                    #     "Score EACH candidate trial against the review question using PICO eligibility: "
                    #     "population/condition, intervention, comparator/control, and study design.\n"
                    #     "Score for likely inclusion, not general topical relevance.\n"
                    #     "Scores: "
                    #     "3=strong candidate for inclusion; the condition/population matches and the intervention/comparator or drug class clearly matches the review question. "
                    #     "Use 3 only when the trial should be protected into the final context. "
                    #     "2=possible candidate; related and may match, but one major PICO element is missing, ambiguous, broader than the review, or only partly aligned. "
                    #     "1=related but probably excluded; shares disease or treatment terms, but likely wrong population, intervention, comparator, or study type. "
                    #     "0=not relevant.\n"
                    #     "Important: most keyword-matching trials should NOT automatically receive 3. "
                    #     "If the trial is merely about the same disease or a nearby drug/intervention, score 1 or 2. "
                    #     "If unsure between 2 and 3, use 3 only when excluding it would likely lose an eligible randomized/interventional trial.\n"
                    #     "Do not downgrade only because clinical results are missing.\n"
                    #     f"Review question:\n{rerank_query}\n\n"
                    #     "Return JSON only. Do not include markdown, prose, summaries, or explanations outside JSON.\n"
                    #     "Use exactly this schema:\n"
                    #     '{"trials":[{"nct_id":"NCT########","score":0,"label":"INCLUDE|MAYBE|EXCLUDE"}]}\n\n'
                    #     "Candidate trials:\n"
                    #     + "\n\n---\n\n".join(cards)
                    # )
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a careful clinical-trial eligibility ranker. "
                                "Return JSON only, with no prose and no markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    response = await create_chat_completion(
                        messages=messages,
                        llm_provider=self.researcher.cfg.strategic_llm_provider,
                        model=self.researcher.cfg.strategic_llm_model,
                        temperature=0,
                        reasoning_effort=ReasoningEfforts.Medium.value,
                        max_tokens=max_tokens,
                        cost_callback=self.researcher.add_costs,
                    )
                    raw_responses.append(response)
                    parsed_scores = self._parse_llm_nct_score_response(response)
                    for nct_id, info in parsed_scores.items():
                        info = dict(info)
                        info["batch_start"] = batch_start
                        score_info[nct_id] = info

                def rank_key(candidate: dict):
                    nct_id = candidate.get("nct_id")
                    info = score_info.get(nct_id, {})
                    score = int(info.get("score", -1))
                    try:
                        best_rank = int(candidate.get("best_retrieval_rank") or 999999)
                    except (TypeError, ValueError):
                        best_rank = 999999
                    try:
                        hit_count = int(candidate.get("retrieval_hit_count") or 0)
                    except (TypeError, ValueError):
                        hit_count = 0
                    try:
                        search_score = float(candidate.get("best_search_score") or 0.0)
                    except (TypeError, ValueError):
                        search_score = 0.0
                    return (
                        -score,
                        best_rank,
                        -hit_count,
                        -search_score,
                        str(nct_id or ""),
                    )

                ranked_candidates = sorted(llm_candidates, key=rank_key)
                for candidate in ranked_candidates:
                    info = score_info.get(candidate.get("nct_id"), {})
                    if info:
                        candidate["llm_rerank_score"] = int(info.get("score", -1))

                selected = ranked_candidates[:keep_limit]

                if len(selected) < min_keep:
                    selected_ids_for_fill = {candidate.get("nct_id") for candidate in selected}
                    for candidate in keyed_candidates:
                        if candidate.get("nct_id") in selected_ids_for_fill:
                            continue
                        selected.append(candidate)
                        selected_ids_for_fill.add(candidate.get("nct_id"))
                        if len(selected) >= min_keep:
                            break

                selected_ids = [candidate.get("nct_id") for candidate in selected if candidate.get("nct_id")]
                selected_ncts = selected_ids
                boosted = [candidate for candidate in selected if int(score_info.get(candidate.get("nct_id"), {}).get("score", -1)) >= 2]

            else:
                cards = [
                    f"[{idx}]\n{self._compact_trial_card_for_llm_rerank(candidate)}"
                    for idx, candidate in enumerate(llm_candidates, start=1)
                ]
                prompt = (
                    "You are selecting ClinicalTrials.gov records for a Cochrane-style systematic review.\n"
                    "Your job is to identify candidate NCT IDs that should be prioritized, not to write a summary.\n"
                    "Classify each candidate trial as INCLUDE, MAYBE, or EXCLUDE for the review question.\n"
                    "Be recall-oriented: prefer MAYBE over EXCLUDE when uncertain.\n"
                    "Use trial eligibility signals: population/condition, intervention, comparator, study design, and outcomes.\n"
                    "Do not require clinical results to be present; registry design information can still be relevant.\n"
                    f"Review question:\n{rerank_query}\n\n"
                    "Return JSON only. Do not include markdown. Do not summarize. Do not explain outside JSON.\n"
                    "Use exactly this schema:\n"
                    '{"trials":[{"nct_id":"NCT########","label":"INCLUDE|MAYBE|EXCLUDE"}]}\n\n'
                    "Candidate trials:\n"
                    + "\n\n---\n\n".join(cards)
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful clinical-trial inclusion classifier. "
                            "Return JSON only, with no prose and no markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                response = await create_chat_completion(
                    messages=messages,
                    llm_provider=self.researcher.cfg.strategic_llm_provider,
                    model=self.researcher.cfg.strategic_llm_model,
                    temperature=0,
                    reasoning_effort=ReasoningEfforts.Medium.value,
                    max_tokens=max_tokens,
                    cost_callback=self.researcher.add_costs,
                )
                raw_responses.append(response)
                selected_ncts, labels = self._parse_llm_nct_rerank_response(response)

                selected_set = set(selected_ncts)
                boosted = [candidate for candidate in keyed_candidates if candidate.get("nct_id") in selected_set]
                boosted = boosted[:keep_limit]
                boosted_ids = {candidate.get("nct_id") for candidate in boosted}

                if rerank_mode == "filter":
                    selected = list(boosted)
                    if len(selected) < min_keep:
                        selected_ids_for_fill = {candidate.get("nct_id") for candidate in selected}
                        for candidate in keyed_candidates:
                            if candidate.get("nct_id") in selected_ids_for_fill:
                                continue
                            selected.append(candidate)
                            selected_ids_for_fill.add(candidate.get("nct_id"))
                            if len(selected) >= min_keep:
                                break
                else:
                    # Boost mode protects recall: LLM-selected trials move first,
                    # but non-selected deduped trials remain available to compression.
                    selected = boosted + [
                        candidate
                        for candidate in keyed_candidates
                        if candidate.get("nct_id") not in boosted_ids
                    ]

                if total_limit > 0:
                    selected = selected[:total_limit]
                selected_ids = [candidate.get("nct_id") for candidate in selected if candidate.get("nct_id")]

        except Exception as exc:
            self.logger.error("LLM_NCT_RERANK failed; falling back to deduped candidates: %s", exc, exc_info=True)
            return candidates

        self.logger.info(
            "LLM_NCT_RERANK mode=%s scored/boosted %d/%d keyed candidates; returning %d candidates",
            rerank_mode,
            len(boosted),
            len(keyed_candidates),
            len(selected),
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "llm_nct_rerank",
                (
                    f"🧪 LLM NCT selector mode={rerank_mode} selected {len(selected)} "
                    f"of {len(keyed_candidates)} deduped trial candidates."
                ),
                self.researcher.websocket,
            )

        self._save_llm_nct_rerank_debug(
            {
                "query": query,
                "rerank_query": rerank_query,
                "rerank_query_source": (
                    "env"
                    if os.getenv("LLM_NCT_RERANK_QUERY", "").strip()
                    else "original_research_query"
                    if str(getattr(self.researcher, "original_research_query", "") or "").strip()
                    else "current_query"
                ),
                "input_limit": input_limit,
                "keep_limit": keep_limit,
                "min_keep": min_keep,
                "total_limit": total_limit,
                "batch_size": batch_size,
                "rrf_k": rrf_k,
                "input_selection_strategy": "stratified_by_subquery_then_best_rank",
                "ranking_strategy": "llm_score_then_best_rank_then_hit_count",
                "rerank_mode": rerank_mode,
                "input_candidate_count": len(llm_candidates),
                "deduped_keyed_candidate_count": len(keyed_candidates),
                "boosted_count": len(boosted),
                "boosted_nct_ids": [candidate.get("nct_id") for candidate in boosted if candidate.get("nct_id")],
                "selected_count": len(selected),
                "selected_nct_ids": selected_ids,
                "llm_labels": labels,
                "llm_scores": score_info,
                "retrieval_rrf": {
                    candidate.get("nct_id"): {
                        "score": candidate.get("retrieval_rrf_score", 0.0),
                        "ranks_by_query": candidate.get("retrieval_ranks_by_query", {}),
                    }
                    for candidate in llm_candidates
                    if candidate.get("nct_id")
                },
                "raw_responses": raw_responses,
            }
        )

        return selected + unkeyed_candidates

    def _dedupe_candidates_by_nct(self, pages: list[dict]) -> list[dict]:
        by_nct: dict[str, dict] = {}
        unkeyed: list[dict] = []

        for page in pages or []:
            if not isinstance(page, dict):
                continue

            nct_id = self._extract_nct_id_from_page(page)
            rank = int(page.get("retrieval_rank") or 999999)
            query = page.get("retrieved_by_query")

            if not nct_id:
                unkeyed.append(dict(page))
                continue

            existing = by_nct.get(nct_id)
            if existing is None:
                item = dict(page)
                item["nct_id"] = nct_id
                item["retrieved_by_queries"] = [query] if query else []
                rank_key = query or "__unknown_1"
                item["retrieval_ranks_by_query"] = {rank_key: rank}
                item["retrieval_rrf_score"] = self._retrieval_rrf_score([rank])
                item["best_retrieval_rank"] = rank
                item["retrieval_hit_count"] = 1
                item["best_search_score"] = self._score_from_page(page)
                by_nct[nct_id] = item
                continue

            existing["retrieval_hit_count"] = existing.get("retrieval_hit_count", 1) + 1
            ranks_by_query = existing.setdefault("retrieval_ranks_by_query", {})
            rank_key = query or f"__unknown_{len(ranks_by_query) + 1}"
            ranks_by_query[rank_key] = min(ranks_by_query.get(rank_key, rank), rank)
            existing["retrieval_rrf_score"] = self._retrieval_rrf_score(
                ranks_by_query.values()
            )
            existing["best_retrieval_rank"] = min(
                existing.get("best_retrieval_rank", 999999),
                rank,
            )
            existing["best_search_score"] = max(
                existing.get("best_search_score", 0.0),
                self._score_from_page(page),
            )

            if query and query not in existing.get("retrieved_by_queries", []):
                existing.setdefault("retrieved_by_queries", []).append(query)

            # Keep the richest content variant for compression.
            if len(page.get("raw_content", "") or "") > len(existing.get("raw_content", "") or ""):
                existing["raw_content"] = page.get("raw_content", "")

        deduped = sorted(
            by_nct.values(),
            key=lambda page: (
                -page.get("retrieval_hit_count", 0),
                page.get("best_retrieval_rank", 999999),
                -page.get("best_search_score", 0.0),
            ),
        )

        max_candidates = int(os.getenv("SUBQUERY_NCT_DEDUPE_MAX_CANDIDATES", "0") or 0)
        if max_candidates > 0:
            deduped = deduped[:max_candidates]

        return deduped + unkeyed

    async def _retrieve_sub_query_candidates(self, sub_query: str, query_domains: list | None = None):
        if query_domains is None:
            query_domains = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        pages = await self._scrape_data_by_urls(sub_query, query_domains)
        annotated = []
        for rank, page in enumerate(pages or [], start=1):
            if not isinstance(page, dict):
                continue
            item = dict(page)
            item["retrieved_by_query"] = sub_query
            item["retrieval_rank"] = rank
            annotated.append(item)

        self.logger.info(
            f"Retrieved {len(annotated)} raw candidates for dedupe sub-query: {sub_query}"
        )
        return annotated

    async def _get_context_by_subquery_nct_dedupe(
        self,
        query: str,
        sub_queries: list[str],
        query_domains: list | None = None,
    ):
        if query_domains is None:
            query_domains = []

        candidate_batches = await asyncio.gather(
            *[
                self._retrieve_sub_query_candidates(sub_query, query_domains)
                for sub_query in sub_queries
            ]
        )
        all_candidates = [
            candidate
            for batch in candidate_batches
            for candidate in batch
        ]
        deduped_candidates = self._dedupe_candidates_by_nct(all_candidates)
        deduped_candidates = await self._llm_rerank_nct_candidates(query, deduped_candidates)

        nct_count = sum(1 for item in deduped_candidates if item.get("nct_id"))
        self.logger.info(
            "SUBQUERY_NCT_DEDUPE pooled %d candidates into %d candidates (%d keyed NCTs)",
            len(all_candidates),
            len(deduped_candidates),
            nct_count,
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subquery_nct_dedupe",
                (
                    f"🧬 Global NCT dedupe pooled {len(all_candidates)} candidates "
                    f"into {len(deduped_candidates)} candidates ({nct_count} NCTs)."
                ),
                self.researcher.websocket,
            )

        if not deduped_candidates:
            return ""

        if self._trial_level_context_enabled():
            trial_context = self._format_trial_level_context(query, deduped_candidates)
            self._save_web_context_debug(query, deduped_candidates, trial_context)
            self.logger.info(
                "Trial-level context produced after SUBQUERY_NCT_DEDUPE: %d chars",
                len(str(trial_context)) if trial_context else 0,
            )
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "trial_level_context",
                    (
                        "Trial-level context enabled: bypassing embedding compression "
                        "and returning compact reranked trial summaries."
                    ),
                    self.researcher.websocket,
                )
            return trial_context

        web_context = await self.researcher.context_manager.get_similar_content_by_query(
            query,
            deduped_candidates,
        )
        self._save_web_context_debug(query, deduped_candidates, web_context)
        self.logger.info(
            f"Web content found after SUBQUERY_NCT_DEDUPE: {len(str(web_context)) if web_context else 0} chars"
        )
        return web_context

    async def _execute_mcp_research_for_queries(self, queries: list, mcp_retrievers: list) -> list:
        """
        Execute MCP research for a list of queries.
        
        Args:
            queries: List of queries to research
            mcp_retrievers: List of MCP retriever classes
            
        Returns:
            list: Combined MCP context entries from all queries
        """
        all_mcp_context = []
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"Executing MCP research for query {i}/{len(queries)}: {query}")
            
            for retriever in mcp_retrievers:
                try:
                    mcp_results = await self._execute_mcp_research(retriever, query)
                    if mcp_results:
                        for result in mcp_results:
                            content = result.get("body", "")
                            url = result.get("href", "")
                            title = result.get("title", "")
                            
                            if content:
                                context_entry = {
                                    "content": content,
                                    "url": url,
                                    "title": title,
                                    "query": query,
                                    "source_type": "mcp"
                                }
                                all_mcp_context.append(context_entry)
                        
                        self.logger.info(f"Added {len(mcp_results)} MCP results for query: {query}")
                        
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results_cached",
                                f"✅ Cached {len(mcp_results)} MCP results from query {i}/{len(queries)}",
                                self.researcher.websocket,
                            )
                except Exception as e:
                    self.logger.error(f"Error in MCP research for query '{query}': {e}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_error",
                            f"⚠️ MCP research error for query {i}, continuing with other sources",
                            self.researcher.websocket,
                        )
        
        return all_mcp_context

    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """Takes in a sub query and scrapes urls based on it and gathers context."""
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            # Identify MCP retrievers
            mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
            non_mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" not in r.__name__.lower()]
            
            # Initialize context components
            mcp_context = []
            web_context = ""
            
            # Get MCP strategy configuration
            mcp_strategy = self._get_mcp_strategy()
            
            # **CONFIGURABLE MCP PROCESSING**
            if mcp_retrievers:
                if mcp_strategy == "disabled":
                    # MCP disabled - skip entirely
                    self.logger.info(f"MCP disabled for sub-query: {sub_query}")
                elif mcp_strategy == "fast" and self._mcp_results_cache is not None:
                    # Fast: Use cached results
                    mcp_context = self._mcp_results_cache.copy()
                    
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_reuse",
                            f"♻️ Reusing cached MCP results ({len(mcp_context)} sources) for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    self.logger.info(f"Reused {len(mcp_context)} cached MCP results for sub-query: {sub_query}")
                elif mcp_strategy == "deep":
                    # Deep: Run MCP for every sub-query
                    self.logger.info(f"Running deep MCP research for: {sub_query}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive_run",
                            f"🔍 Running deep MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
                else:
                    # Fallback: if no cache and not deep mode, run MCP for this query
                    self.logger.warning("MCP cache not available, falling back to per-sub-query execution")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_fallback",
                            f"🔌 MCP cache unavailable, running MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
            
            # Get web search context using non-MCP retrievers (if no scraped data provided)
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            # Get similar content based on scraped data
            if scraped_data:
                web_context = await self.researcher.context_manager.get_similar_content_by_query(sub_query, scraped_data)
                self._save_web_context_debug(sub_query, scraped_data, web_context)
                self.logger.info(f"Web content found for sub-query: {len(str(web_context)) if web_context else 0} chars")

            # Combine MCP context with web context intelligently
            combined_context = self._combine_mcp_and_web_context(mcp_context, web_context, sub_query)
            
            # Log context combination results
            if combined_context:
                context_length = len(str(combined_context))
                self.logger.info(f"Combined context for '{sub_query}': {context_length} chars")
                
                if self.researcher.verbose:
                    mcp_count = len(mcp_context)
                    web_available = bool(web_context)
                    cache_used = self._mcp_results_cache is not None and mcp_retrievers and mcp_strategy != "deep"
                    cache_status = " (cached)" if cache_used else ""
                    await stream_output(
                        "logs",
                        "context_combined",
                        f"📚 Combined research context: {mcp_count} MCP sources{cache_status}, {'web content' if web_available else 'no web content'}",
                        self.researcher.websocket,
                    )
            else:
                self.logger.warning(f"No combined context found for sub-query: {sub_query}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_not_found",
                        f"🤷 No content found for '{sub_query}'...",
                        self.researcher.websocket,
                    )
            
            if combined_context and self.json_handler:
                self.json_handler.log_event("content_found", {
                    "sub_query": sub_query,
                    "content_size": len(str(combined_context)),
                    "mcp_sources": len(mcp_context),
                    "web_content": bool(web_context)
                })
                
            return combined_context
            
        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_error",
                    f"❌ Error processing '{sub_query}': {str(e)}",
                    self.researcher.websocket,
                )
            return ""

    async def _execute_mcp_research(self, retriever, query):
        """
        Execute MCP research using the new two-stage approach.
        
        Args:
            retriever: The MCP retriever class
            query: The search query
            
        Returns:
            list: MCP research results
        """
        retriever_name = retriever.__name__
        
        self.logger.info(f"Executing MCP research with {retriever_name} for query: {query}")
        
        try:
            # Instantiate the MCP retriever with proper parameters
            # Pass the researcher instance (self.researcher) which contains both cfg and mcp_configs
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket,
                researcher=self.researcher  # Pass the entire researcher instance
            )
            
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval_stage1",
                    f"🧠 Stage 1: Selecting optimal MCP tools for: {query}",
                    self.researcher.websocket,
                )
            
            # Execute the two-stage MCP search
            results = retriever_instance.search(
                max_results=self.researcher.cfg.max_search_results_per_query
            )
            
            if results:
                result_count = len(results)
                self.logger.info(f"MCP research completed: {result_count} results from {retriever_name}")
                
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_research_complete",
                        f"🎯 MCP research completed: {result_count} intelligent results obtained",
                        self.researcher.websocket,
                    )
                
                return results
            else:
                self.logger.info(f"No results returned from MCP research with {retriever_name}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_no_results",
                        f"ℹ️ No relevant information found via MCP for: {query}",
                        self.researcher.websocket,
                    )
                return []
                
        except Exception as e:
            self.logger.error(f"Error in MCP research with {retriever_name}: {str(e)}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_research_error",
                    f"⚠️ MCP research error: {str(e)} - continuing with other sources",
                    self.researcher.websocket,
                )
            return []

    def _combine_mcp_and_web_context(self, mcp_context: list, web_context: str, sub_query: str) -> str:
        """
        Intelligently combine MCP and web research context.
        
        Args:
            mcp_context: List of MCP context entries
            web_context: Web research context string  
            sub_query: The sub-query being processed
            
        Returns:
            str: Combined context string
        """
        combined_parts = []
        
        # Add web context first if available
        if web_context and web_context.strip():
            combined_parts.append(web_context.strip())
            self.logger.debug(f"Added web context: {len(web_context)} chars")
        
        # Add MCP context with proper formatting
        if mcp_context:
            mcp_formatted = []
            
            for i, item in enumerate(mcp_context):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"MCP Result {i+1}")
                
                if content and content.strip():
                    # Create a well-formatted context entry
                    if url and url != f"mcp://llm_analysis":
                        citation = f"\n\n*Source: {title} ({url})*"
                    else:
                        citation = f"\n\n*Source: {title}*"
                    
                    formatted_content = f"{content.strip()}{citation}"
                    mcp_formatted.append(formatted_content)
            
            if mcp_formatted:
                # Join MCP results with clear separation
                mcp_section = "\n\n---\n\n".join(mcp_formatted)
                combined_parts.append(mcp_section)
                self.logger.debug(f"Added {len(mcp_context)} MCP context entries")
        
        # Combine all parts
        if combined_parts:
            final_context = "\n\n".join(combined_parts)
            self.logger.info(f"Combined context for '{sub_query}': {len(final_context)} total chars")
            return final_context
        else:
            self.logger.warning(f"No context to combine for sub-query: {sub_query}")
            return ""

    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: dict | None = None):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        context = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        return context

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "added_source_url",
                        f"✅ Added source url to research: {url}\n",
                        self.researcher.websocket,
                        True,
                        url,
                    )

        return new_urls

    async def _search_relevant_source_urls(self, query, query_domains: list | None = None):
        new_search_urls = []
        prefetched_content = []
        if query_domains is None:
            query_domains = []

        # Iterate through the currently set retrievers
        # This allows the method to work when retrievers are temporarily modified
        for retriever_class in self.researcher.retrievers:
            # Skip MCP retrievers as they don't provide URLs for scraping
            if "mcpretriever" in retriever_class.__name__.lower():
                continue

            try:
                # Instantiate the retriever with the sub-query
                retriever = retriever_class(query, query_domains=query_domains)

                # Perform the search using the current retriever
                search_results = await asyncio.to_thread(
                    retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
                )

                if not search_results:
                    continue

                # Separate results that already have content from those needing scraping
                for result in search_results:
                    url = result.get("href") or result.get("url")
                    raw_content = result.get("raw_content")
                    if url and raw_content and len(raw_content) > 100:
                        # Only raw_content signals that a retriever already fetched the full page.
                        # body is snippet-sized text for most web retrievers and still needs scraping.
                        item = dict(result)
                        item["url"] = url
                        item["raw_content"] = raw_content
                        prefetched_content.append(item)
                        self.researcher.add_research_sources([{"url": url}])
                    elif url:
                        new_search_urls.append(url)
            except Exception as e:
                self.logger.error(f"Error searching with {retriever_class.__name__}: {e}")

        # Get unique URLs
        new_search_urls = await self._get_new_urls(new_search_urls)
        random.shuffle(new_search_urls)

        return new_search_urls, prefetched_content

    async def _scrape_data_by_urls(self, sub_query, query_domains: list | None = None):
        """
        Runs a sub-query across multiple retrievers and scrapes the resulting URLs.
        Retrievers that already provide full content (e.g. PubMed Central) have their
        content passed through directly without re-scraping.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        if query_domains is None:
            query_domains = []

        new_search_urls, prefetched_content = await self._search_relevant_source_urls(sub_query, query_domains)

        # Log the research process if verbose mode is on
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                f"🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # Scrape URLs that need fetching (skip those already provided by retrievers)
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)

        # Merge pre-fetched content from retrievers that already provide full text
        scraped_content.extend(prefetched_content)

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content

    async def _search(self, retriever, query):
        """
        Perform a search using the specified retriever.
        
        Args:
            retriever: The retriever class to use
            query: The search query
            
        Returns:
            list: Search results
        """
        retriever_name = retriever.__name__
        is_mcp_retriever = "mcpretriever" in retriever_name.lower()
        
        self.logger.info(f"Searching with {retriever_name} for query: {query}")
        
        try:
            # Instantiate the retriever
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket if is_mcp_retriever else None,
                researcher=self.researcher if is_mcp_retriever else None
            )
            
            # Log MCP server configurations if using MCP retriever
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval",
                    f"🔌 Consulting MCP server(s) for information on: {query}",
                    self.researcher.websocket,
                )
            
            # Perform the search
            if hasattr(retriever_instance, 'search'):
                results = retriever_instance.search(
                    max_results=self.researcher.cfg.max_search_results_per_query
                )
                
                # Log result information
                if results:
                    result_count = len(results)
                    self.logger.info(f"Received {result_count} results from {retriever_name}")
                    
                    # Special logging for MCP retriever
                    if is_mcp_retriever:
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results",
                                f"✓ Retrieved {result_count} results from MCP server",
                                self.researcher.websocket,
                            )
                        
                        # Log result details
                        for i, result in enumerate(results[:3]):  # Log first 3 results
                            title = result.get("title", "No title")
                            url = result.get("href", "No URL")
                            content_length = len(result.get("body", "")) if result.get("body") else 0
                            self.logger.info(f"MCP result {i+1}: '{title}' from {url} ({content_length} chars)")
                            
                        if result_count > 3:
                            self.logger.info(f"... and {result_count - 3} more MCP results")
                else:
                    self.logger.info(f"No results returned from {retriever_name}")
                    if is_mcp_retriever and self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_no_results",
                            f"ℹ️ No relevant information found from MCP server for: {query}",
                            self.researcher.websocket,
                        )
                
                return results
            else:
                self.logger.error(f"Retriever {retriever_name} does not have a search method")
                return []
        except Exception as e:
            self.logger.error(f"Error searching with {retriever_name}: {str(e)}")
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_error",
                    f"❌ Error retrieving information from MCP server: {str(e)}",
                    self.researcher.websocket,
                )
            return []
            
    async def _extract_content(self, results):
        """
        Extract content from search results using the browser manager.
        
        Args:
            results: Search results
            
        Returns:
            list: Extracted content
        """
        self.logger.info(f"Extracting content from {len(results)} search results")
        
        # Get the URLs from the search results
        urls = []
        for result in results:
            if isinstance(result, dict) and "href" in result:
                urls.append(result["href"])
        
        # Skip if no URLs found
        if not urls:
            return []
            
        # Make sure we don't visit URLs we've already visited
        new_urls = [url for url in urls if url not in self.researcher.visited_urls]
        
        # Return empty if no new URLs
        if not new_urls:
            return []
            
        # Scrape the content from the URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_urls)
        
        # Add the URLs to visited_urls
        self.researcher.visited_urls.update(new_urls)
        
        return scraped_content
        
    async def _summarize_content(self, query, content):
        """
        Summarize the extracted content.
        
        Args:
            query: The search query
            content: The extracted content
            
        Returns:
            str: Summarized content
        """
        self.logger.info(f"Summarizing content for query: {query}")
        
        # Skip if no content
        if not content:
            return ""
            
        # Summarize the content using the context manager
        summary = await self.researcher.context_manager.get_similar_content_by_query(
            query, content
        )
        
        return summary
        
    async def _update_search_progress(self, current, total):
        """
        Update the search progress.
        
        Args:
            current: Current number of sub-queries processed
            total: Total number of sub-queries
        """
        if self.researcher.verbose and self.researcher.websocket:
            progress = int((current / total) * 100)
            await stream_output(
                "logs",
                "research_progress",
                f"📊 Research Progress: {progress}%",
                self.researcher.websocket,
                True,
                {
                    "current": current,
                    "total": total,
                    "progress": progress
                }
            )
