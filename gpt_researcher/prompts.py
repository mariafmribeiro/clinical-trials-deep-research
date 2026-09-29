import warnings
from datetime import date, datetime, timezone

from langchain_core.documents import Document

from .config import Config
from .utils.enum import ReportSource, ReportType, Tone
from .utils.enum import PromptFamily as PromptFamilyEnum
from typing import Callable, List, Dict, Any


## Prompt Families #############################################################

class PromptFamily:
    """General purpose class for prompt formatting.

    This may be overwritten with a derived class that is model specific. The
    methods are broken down into two groups:

    1. Prompt Generators: These follow a standard format and are correlated with
        the ReportType enum. They should be accessed via
        get_prompt_by_report_type

    2. Prompt Methods: These are situation-specific methods that do not have a
        standard signature and are accessed directly in the agent code.

    All derived classes must retain the same set of method names, but may
    override individual methods.
    """

    def __init__(self, config: Config):
        """Initialize with a config instance. This may be used by derived
        classes to select the correct prompting based on configured models and/
        or providers
        """
        self.cfg = config

    # MCP-specific prompts
    @staticmethod
    def generate_mcp_tool_selection_prompt(query: str, tools_info: List[Dict], max_tools: int = 3) -> str:
        """
        Generate prompt for LLM-based MCP tool selection.
        
        Args:
            query: The research query
            tools_info: List of available tools with their metadata
            max_tools: Maximum number of tools to select
            
        Returns:
            str: The tool selection prompt
        """
        import json
        
        return f"""You are a research assistant helping to select the most relevant tools for a research query.

RESEARCH QUERY: "{query}"

AVAILABLE TOOLS:
{json.dumps(tools_info, indent=2)}

TASK: Analyze the tools and select EXACTLY {max_tools} tools that are most relevant for researching the given query.

SELECTION CRITERIA:
- Choose tools that can provide information, data, or insights related to the query
- Prioritize tools that can search, retrieve, or access relevant content
- Consider tools that complement each other (e.g., different data sources)
- Exclude tools that are clearly unrelated to the research topic

Return a JSON object with this exact format:
{{
  "selected_tools": [
    {{
      "index": 0,
      "name": "tool_name",
      "relevance_score": 9,
      "reason": "Detailed explanation of why this tool is relevant"
    }}
  ],
  "selection_reasoning": "Overall explanation of the selection strategy"
}}

Select exactly {max_tools} tools, ranked by relevance to the research query.
"""

    @staticmethod
    def generate_mcp_research_prompt(query: str, selected_tools: List) -> str:
        """
        Generate prompt for MCP research execution with selected tools.
        
        Args:
            query: The research query
            selected_tools: List of selected MCP tools
            
        Returns:
            str: The research execution prompt
        """
        # Handle cases where selected_tools might be strings or objects with .name attribute
        tool_names = []
        for tool in selected_tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            else:
                tool_names.append(str(tool))
        
        return f"""You are a research assistant with access to specialized tools. Your task is to research the following query and provide comprehensive, accurate information.

RESEARCH QUERY: "{query}"

INSTRUCTIONS:
1. Use the available tools to gather relevant information about the query
2. Call multiple tools if needed to get comprehensive coverage
3. If a tool call fails or returns empty results, try alternative approaches
4. Synthesize information from multiple sources when possible
5. Focus on factual, relevant information that directly addresses the query

AVAILABLE TOOLS: {tool_names}

Please conduct thorough research and provide your findings. Use the tools strategically to gather the most relevant and comprehensive information."""

    # Image generation prompts
    @staticmethod
    def generate_image_analysis_prompt(
        query: str,
        sections: List[Dict[str, Any]],
        max_images: int = 3,
    ) -> str:
        """Generate prompt for analyzing which report sections need images.
        
        Args:
            query: The research query.
            sections: List of report sections with header and content.
            max_images: Maximum number of images to suggest.
            
        Returns:
            str: The analysis prompt.
        """
        sections_text = "\n\n".join([
            f"### Section {i+1}: {s['header']}\n{s['content'][:500]}..."
            for i, s in enumerate(sections)
        ])
        
        return f"""Analyze the following research report sections and identify which {max_images} sections would benefit MOST from a visual illustration or diagram.

RESEARCH TOPIC: {query}

REPORT SECTIONS:
{sections_text}

For each recommended section, provide:
1. The section number (1-indexed)
2. A specific, detailed image prompt that would create an informative illustration
3. A brief explanation of why this section benefits from visualization

IMPORTANT GUIDELINES:
- Choose sections where visual representation would genuinely aid understanding
- Focus on concepts, processes, comparisons, data flows, or statistics that are inherently visual
- Avoid sections that are purely textual analysis, introductions, or conclusions
- The image prompt should be specific enough to generate a relevant, professional illustration
- Images should be informative and educational, not decorative
- Consider diagrams, flowcharts, comparison charts, or conceptual illustrations

Respond in JSON format:
{{
    "suggestions": [
        {{
            "section_number": 1,
            "section_header": "Section Title",
            "image_prompt": "Detailed prompt for generating an informative illustration...",
            "image_type": "diagram|flowchart|comparison|concept|data_visualization",
            "reason": "Why this section benefits from visualization"
        }}
    ]
}}

Return ONLY the JSON, no additional text."""

    @staticmethod
    def generate_image_prompt_enhancement(
        base_prompt: str,
        section_content: str,
        research_topic: str,
    ) -> str:
        """Enhance an image prompt with context for better generation.
        
        Args:
            base_prompt: The base image generation prompt.
            section_content: Content from the report section.
            research_topic: The main research topic.
            
        Returns:
            str: Enhanced image prompt.
        """
        return f"""Create a professional, informative illustration for a research report.

RESEARCH TOPIC: {research_topic}

IMAGE DESCRIPTION: {base_prompt}

CONTEXT FROM REPORT:
{section_content[:800]}

STYLE REQUIREMENTS:
- Professional and clean design suitable for academic/business reports
- Clear, easy-to-understand visual elements
- Modern, minimalist aesthetic
- Use a professional color palette (blues, teals, grays)
- Avoid excessive text in the image
- High contrast for readability
- If showing data or comparisons, use clear labels and legends
- Suitable for both digital viewing and printing"""

    @staticmethod
    def generate_search_queries_prompt(
        question: str,
        parent_query: str,
        report_type: str,
        max_iterations: int = 3,
        context: List[Dict[str, Any]] = [],
    ):
        """Generates the search queries prompt for the given question.
        Args:
            question (str): The question to generate the search queries prompt for
            parent_query (str): The main question (only relevant for detailed reports)
            report_type (str): The report type
            max_iterations (int): The maximum number of search queries to generate
            context (str): Context for better understanding of the task with realtime web information

        Returns: str: The search queries prompt for the given question
        """

        if (
            report_type == ReportType.DetailedReport.value
            or report_type == ReportType.SubtopicReport.value
        ):
            task = f"{parent_query} - {question}"
        else:
            task = question

        # FIX: this used to be built and then never actually included in the
        # returned prompt below, so the preliminary search plan_research()
        # performs (and deliberately truncates, see PLAN_RESEARCH_CONTEXT_CHARS
        # in researcher.py) was silently discarded. Wording adapted from the
        # original web-research phrasing ("real-time web information", "current
        # events") to this local, static CTR corpus, and pointed specifically at
        # closing the class-term-vs-specific-name gap (Section 5/6 hotspot
        # analysis): a review question phrased at the drug-class level ("TKI
        # therapy") often won't lexically match a trial named by its specific
        # compound ("sunitinib"), and this preliminary context is exactly what
        # can surface that compound name before query generation runs.
        context_prompt = f"""
You are a seasoned research assistant tasked with generating search queries to find relevant information for the following task: "{task}".
Preliminary search results already retrieved for this task: {context}

Use these results to inform and refine your search queries: prefer concrete terms that actually appear in these retrieved records, such as specific intervention or drug names, over the more general terms in the task itself, when they refer to the same thing.
""" if context else ""

        dynamic_example = ", ".join([f'"query {i+1}"' for i in range(max_iterations)])

        return f"""Write {max_iterations} search queries for a local OpenSearch index of ClinicalTrials.gov trial records from the TREC Clinical Trials 2023 dataset.

            Task: "{task}"
            {context_prompt}
            Important context:
            - The search source is NOT the web.
            - The search source is a local index of ClinicalTrials.gov XML trial records.
            - The dataset only contains records available up to 2023.
            - Do NOT generate Google-style queries.
            - Do NOT include words like "latest", "recent", "news", "review", "meta-analysis", "Cochrane", "PubMed", or "guideline" unless they are part of the user's original question.
            - Prefer compact clinical search queries using condition, population, intervention, comparator, and outcome terms.
            - Use synonyms when useful, but keep each query short.

            You must respond with a list of strings in the following format: [{dynamic_example}].
            The response should contain ONLY the list.
            """

    @staticmethod
    def generate_report_prompt(
        question: str,
        context,
        report_source: str,
        report_format="apa",
        total_words=1000,
        tone=None,
        language="english",
    ):
        """Generates the report prompt for the given question and research summary.
        Args: question (str): The question to generate the report prompt for
                research_summary (str): The research summary to generate the report prompt for
        Returns: str: The report prompt for the given question and research summary
        """

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
"""

        tone_prompt = f"Write the report in a {tone.value} tone." if tone else ""

        return f"""
Information: "{context}"
---
Using the above information, answer the following query or task: "{question}" in a detailed report --
The report should focus on the answer to the query, should be well structured, informative,
in-depth, and comprehensive, with facts and numbers if available and at least {total_words} words.
You should strive to write the report as long as you can using all relevant and necessary information provided.

Please follow all of the following guidelines in your report:
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST write the report with markdown syntax and {report_format} format.
- Structure your report with clear markdown headers: use # for the main title, ## for major sections, and ### for subsections.
- Use markdown tables when presenting structured data or comparisons to enhance readability.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- You MUST NOT include a table of contents, but DO include proper markdown headers (# ## ###) to structure your report clearly.
- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
- Don't forget to add a reference list at the end of the report in {report_format} format and full url links without hyperlinks.
- {reference_prompt}
- {tone_prompt}
You MUST write the report in the following language: {language}.
Please do your best, this is very important to my career.
Assume that the current date is {date.today()}.
"""

    @staticmethod
    def curate_sources(query, sources, max_results=10):
        return f"""Your goal is to evaluate and curate the provided scraped content for the research task: "{query}"
    while prioritizing the inclusion of relevant and high-quality information, especially sources containing statistics, numbers, or concrete data.

The final curated list will be used as context for creating a research report, so prioritize:
- Retaining as much original information as possible, with extra emphasis on sources featuring quantitative data or unique insights
- Including a wide range of perspectives and insights
- Filtering out only clearly irrelevant or unusable content

EVALUATION GUIDELINES:
1. Assess each source based on:
   - Relevance: Include sources directly or partially connected to the research query. Err on the side of inclusion.
   - Credibility: Favor authoritative sources but retain others unless clearly untrustworthy.
   - Currency: Prefer recent information unless older data is essential or valuable.
   - Objectivity: Retain sources with bias if they provide a unique or complementary perspective.
   - Quantitative Value: Give higher priority to sources with statistics, numbers, or other concrete data.
2. Source Selection:
   - Include as many relevant sources as possible, up to {max_results}, focusing on broad coverage and diversity.
   - Prioritize sources with statistics, numerical data, or verifiable facts.
   - Overlapping content is acceptable if it adds depth, especially when data is involved.
   - Exclude sources only if they are entirely irrelevant, severely outdated, or unusable due to poor content quality.
3. Content Retention:
   - DO NOT rewrite, summarize, or condense any source content.
   - Retain all usable information, cleaning up only clear garbage or formatting issues.
   - Keep marginally relevant or incomplete sources if they contain valuable data or insights.

SOURCES LIST TO EVALUATE:
{sources}

You MUST return your response in the EXACT sources JSON list format as the original sources.
The response MUST not contain any markdown format or additional text (like ```json), just the JSON list!
"""

    @staticmethod
    def generate_resource_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None, total_words=1000, language="english"
    ):
        """Generates the resource report prompt for the given question and research summary.

        Args:
            question (str): The question to generate the resource report prompt for.
            context (str): The research summary to generate the resource report prompt for.

        Returns:
            str: The resource report prompt for the given question and research summary.
        """

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
            You MUST include all relevant source urls.
            Every url should be hyperlinked: [url website](url)
            """
        else:
            reference_prompt = f"""
            You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
        """

        return (
            f'"""{context}"""\n\nBased on the above information, generate a bibliography recommendation report for the following'
            f' question or topic: "{question}". The report should provide a detailed analysis of each recommended resource,'
            " explaining how each source can contribute to finding answers to the research question.\n"
            "Focus on the relevance, reliability, and significance of each source.\n"
            "Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax.\n"
            "Use markdown tables and other formatting features when appropriate to organize and present information clearly.\n"
            "Include relevant facts, figures, and numbers whenever available.\n"
            f"The report should have a minimum length of {total_words} words.\n"
            f"You MUST write the report in the following language: {language}.\n"
            "You MUST include all relevant source urls."
            "Every url should be hyperlinked: [url website](url)"
            f"{reference_prompt}"
        )

    @staticmethod
    def generate_custom_report_prompt(
        query_prompt, context, report_source: str, report_format="apa", tone=None, total_words=1000, language: str = "english"
    ):
        return f'"{context}"\n\n{query_prompt}'

    @staticmethod
    def generate_outline_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None,  total_words=1000, language: str = "english"
    ):
        """Generates the outline report prompt for the given question and research summary.
        Args: question (str): The question to generate the outline report prompt for
                research_summary (str): The research summary to generate the outline report prompt for
        Returns: str: The outline report prompt for the given question and research summary
        """

        return (
            f'"""{context}""" Using the above information, generate an outline for a research report in Markdown syntax'
            f' for the following question or topic: "{question}". The outline should provide a well-structured framework'
            " for the research report, including the main sections, subsections, and key points to be covered."
            f" The research report should be detailed, informative, in-depth, and a minimum of {total_words} words."
            " Use appropriate Markdown syntax to format the outline and ensure readability."
            " Consider using markdown tables and other formatting features where they would enhance the presentation of information."
        )

    @staticmethod
    def generate_deep_research_prompt(
        question: str,
        context: str,
        report_source: str,
        report_format="apa",
        tone=None,
        total_words=2000,
        language: str = "english"
    ):
        """Generates the deep research report prompt, specialized for handling hierarchical research results.
        Args:
            question (str): The research question
            context (str): The research context containing learnings with citations
            report_source (str): Source of the research (web, etc.)
            report_format (str): Report formatting style
            tone: The tone to use in writing
            total_words (int): Minimum word count
            language (str): Output language
        Returns:
            str: The deep research report prompt
        """
        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
"""

        tone_prompt = f"Write the report in a {tone.value} tone." if tone else ""

#         return f"""
# You are writing a structured Cochrane-style evidence summary based ONLY on retrieved ClinicalTrials.gov trial records from the local TREC Clinical Trials 2023 dataset.

# Retrieved trial information and citations:
# "{context}"

# Research question:
# "{question}"

# Important constraints:

# - This is NOT live web research.
# - The evidence comes only from local ClinicalTrials.gov XML trial records available up to 2023.
# - Do NOT claim to have searched PubMed, CENTRAL, Embase, Cochrane Library, Google, or the live web.
# - Do NOT describe the answer as a real Cochrane review; call it a Cochrane-style summary based on retrieved registry records.
# - ClinicalTrials.gov records may describe trial design, eligibility, interventions, outcomes, and sometimes results, but may not contain full published results.
# - Do NOT claim effectiveness or safety unless results are explicitly present in the retrieved records.
# - Use cautious certainty wording: "may", "suggests", "is uncertain", "registry evidence was insufficient", or "results were not available".

# Synthesis requirements:
# - Do NOT write the report as a trial-by-trial list.
# - First internally group the retrieved records by intervention class, disease setting, comparator, outcomes, and results availability.
# - In the final answer, write only 3 to 4 concise paragraphs.
# - Each paragraph should synthesize broad conclusions, not describe individual trials one by one.
# - Use individual NCT records only as brief supporting examples when necessary.
# - If the question is broad, cover the main intervention classes, populations, comparators, outcomes, and safety findings represented in the retrieved records.
# - If registry records do not contain enough results, state that briefly, but do not repeat this limitation in every paragraph.


# Output requirements:

# - Write around 700 to 1200 words, unless very little relevant evidence was retrieved.
# - Focus directly on the research question.
# - Include citations using exact NCT IDs from the retrieved context, e.g. (NCT01234567).
# - Never invent NCT IDs, titles, URLs, publication details, PMIDs, or journal references.
# - In the References section, list each cited NCT once as:
#     - NCT ID – Brief title
# - Write in {language}.

# {reference_prompt}

# """
#         return f"""
#         You are writing a structured Cochrane-style evidence summary based ONLY on retrieved ClinicalTrials.gov trial records from the local TREC Clinical Trials 2023 dataset.

#         Retrieved trial information and citations:
#         "{context}"

#         Research question:
#         "{question}"

#         Important constraints:
#         - This is NOT live web research.
#         - The evidence comes only from local ClinicalTrials.gov XML trial records available up to 2023.
#         - Do NOT claim to have searched PubMed, CENTRAL, Embase, Cochrane Library, Google, or the live web.
#         - Do NOT describe the answer as a real Cochrane review; call it a Cochrane-style summary based on retrieved registry records.
#         - ClinicalTrials.gov records may describe trial design, eligibility, interventions, outcomes, and sometimes results, but may not contain full published results.
#         - Do NOT claim effectiveness or safety unless results are explicitly present in the retrieved records.
#         - Use cautious certainty wording: "may", "suggests", "is uncertain", "registry evidence was insufficient", or "results were not available".

#         Synthesis requirements:
#         - Write a compact Cochrane-style conclusion, not a trial-by-trial catalogue.
#         - Start with a direct answer to the research question: whether the intervention appears beneficial, harmful, neutral, or uncertain based on the retrieved registry evidence.
#         - Then explain the main evidence pattern across trials: populations, interventions, comparators, outcomes, and whether results were available.
#         - Use NCT IDs as evidence anchors. When several relevant trials support the same point, cite several NCT IDs together.
#         - Do not cite only one or two representative trials if more relevant NCT IDs in the context support the same conclusion.
#         - If evidence is mixed or incomplete, say so clearly.
#         - If most records contain design information but no results, conclude that registry evidence is insufficient to determine effectiveness/safety.
#         - Avoid overclaiming: do not infer clinical benefit or harm unless explicit results are present.

#         Output requirements:
#         - Keep the narrative concise, around 700 to 1200 words, excluding the ## References section.
#         - The report MUST end with a final markdown section whose heading is exactly:## References
#         - Include exact NCT IDs from the retrieved context in the relevant sentences, e.g. (NCT01234567; NCT04567890).
#         - Every NCT ID mentioned in the report body MUST appear once in ## References.
#         - The References section MUST be present.
#         - In ## References, list each cited NCT once as:
#             - NCT ID – Brief title
#         - Never invent NCT IDs, titles, URLs, publication details, PMIDs, or journal references.
#         - Write in {language}.

# """

#         return f"""
# You are writing a compact Cochrane-style evidence summary based ONLY on retrieved ClinicalTrials.gov trial records from the local TREC Clinical Trials 2023 dataset.

# Retrieved trial information and citations:
# "{context}"

# Research question:
# "{question}"

# Task:
# Answer the research question using the retrieved trial records. Your goal is not only to write a fluent summary, but also to preserve coverage of the retrieved trials that are directly relevant to the question.

# Important constraints:
# - This is NOT live web research.
# - The evidence comes only from local ClinicalTrials.gov XML trial records available up to 2023.
# - Do NOT claim to have searched PubMed, CENTRAL, Embase, Cochrane Library, Google, or the live web.
# - Do NOT describe the answer as a real Cochrane review; call it a Cochrane-style summary based on retrieved registry records.
# - ClinicalTrials.gov records may describe trial design, eligibility, interventions, outcomes, and sometimes results, but may not contain full published results.
# - Do NOT claim effectiveness or safety unless results are explicitly present in the retrieved records.
# - Use cautious certainty wording: "may", "suggests", "is uncertain", "registry evidence was insufficient", or "results were not available".
# - Never invent NCT IDs, titles, URLs, publication details, PMIDs, or journal references.

# How to use the evidence:
# - First identify the NCT IDs in the context that are most directly relevant to the research question.
# - Prefer trials matching the population, intervention, comparator, and outcomes in the question.
# - If reranking signals are present, prioritize trials with higher LLM relevance scores, but do not ignore lower-scored trials if their title/summary clearly matches the question.
# - Do not cite only one or two representative trials when several relevant trials support the same conclusion.
# - When several trials address the same intervention/comparator/outcome, cite the relevant NCT IDs together.
# - If a relevant trial has no results, it can still be cited as registry design evidence, but do not use it to claim benefit or harm.

# Output requirements:
# - Use exactly these sections:
#   ## Conclusion
#   ## Evidence Used
#   ## References
# - In ## Conclusion, give a direct answer: beneficial, harmful, neutral, mixed, or uncertain based on retrieved registry evidence.
# - In ## Evidence Used, write compact grouped bullets. Each bullet should include the relevant NCT IDs and what they contribute.
# - Keep the report concise. Do not write a trial-by-trial catalogue, but do preserve the relevant NCT IDs.
# - Include exact NCT IDs from the retrieved context, e.g. (NCT01234567; NCT04567890).
# - The report MUST end with a final markdown section whose heading is exactly:
#   ## References
# - In ## References, list each cited NCT once as:
#     - NCT ID – Brief title
# - Every NCT ID mentioned in ## Conclusion or ## Evidence Used MUST appear once in ## References.
# - Write in {language}.
# """

#         return f"""
# You are writing a compact Cochrane-style evidence summary based ONLY on retrieved ClinicalTrials.gov trial records from the local TREC Clinical Trials 2023 dataset.

# Retrieved trial information:
# "{context}"

# Research question:
# "{question}"

# Write a concise report that answers the research question based only on the retrieved registry records.

# Rules:
# - Do NOT describe this as a real Cochrane review; call it a Cochrane-style summary based on retrieved registry records.
# - Do NOT claim effectiveness or safety unless explicit results are present in the retrieved records.
# - If results are mostly unavailable, conclude that registry evidence is insufficient.
# - Use cautious wording such as "may", "suggests", "uncertain", or "insufficient evidence".
# - Use exact NCT IDs from the context as citations.
# - Do not invent NCT IDs, titles, publication details, PMIDs, journals, or URLs.

# Evidence coverage:
# - Before writing the conclusion, identify the trials that directly match the population, intervention, comparator, and outcomes in the question.
# - Cite all trials that are directly relevant to the question, not only one or two representative examples.
# - Group multiple NCT IDs together when they support the same point.
# - Exclude trials that are only topically related but clearly study the wrong population, intervention, comparator, or disease setting.

# Output format:
# - Use exactly these sections:
#   ## Summary
#   ## References
# - In ## Summary, write 3 to 5 concise paragraphs.
# - The first paragraph must give the direct answer: beneficial, harmful, neutral, mixed, or uncertain.
# - The remaining paragraphs should synthesize the evidence by intervention/comparator/outcome and cite the relevant NCT IDs.
# - The report MUST end with a final markdown section whose heading is exactly:
#   ## References
# - In ## References, list each cited NCT once as:
#     - NCT ID – Brief title
# - Every NCT ID cited in ## Summary MUST appear once in ## References.
# - Write in {language}.
# """

#         return f"""
# You are writing a Cochrane-style evidence summary based ONLY on retrieved ClinicalTrials.gov trial records from the local TREC Clinical Trials 2023 dataset.

# Retrieved trial information and citations:
# "{context}"

# Research question:
# "{question}"

# Important constraints:
# - This is NOT live web research.
# - The evidence comes only from local ClinicalTrials.gov XML trial records available up to 2023.
# - Do NOT claim to have searched PubMed, CENTRAL, Embase, Cochrane Library, Google, or the live web.
# - Do NOT describe this as a real Cochrane review; call it a Cochrane-style summary based on retrieved registry records.
# - ClinicalTrials.gov records may describe trial design, eligibility, interventions, outcomes, and sometimes results, but may not contain full published results.
# - Do NOT claim effectiveness or safety unless results are explicitly present in the retrieved records.
# - Use cautious certainty wording: "may", "suggests", "is uncertain", "registry evidence was insufficient", or "results were not available".

# Main task:
# Write a synthesis that answers the research question while preserving citation coverage.

# Before writing, identify all NCT IDs in the context that are directly relevant to the research question.
# In the report body, cite every directly relevant NCT ID that supports the synthesis.
# Do not cite only one or two representative examples if more relevant NCT IDs in the context support the same point.
# When several trials support the same point, group their NCT IDs together in one citation, for example: (NCT01234567; NCT04567890; NCT07890123).

# Synthesis requirements:
# - Start with a direct answer to the research question: whether the intervention appears beneficial, harmful, neutral, or uncertain based on the retrieved registry evidence.
# - Then synthesize the evidence by intervention class, comparator, population/setting, outcomes, and results availability.
# - Do NOT write a trial-by-trial catalogue.
# - However, do preserve coverage of the relevant trials by citing their NCT IDs in the appropriate synthesis sentences.
# - If the question is broad, cover the main intervention classes, populations, comparators, outcomes, and safety findings represented in the retrieved records.
# - If evidence is mixed, incomplete, or mostly design-only registry information, say so clearly.
# - If most records contain design information but no results, conclude that registry evidence is insufficient to determine effectiveness or safety.
# - Avoid overclaiming: do not infer clinical benefit or harm unless explicit results are present.

# Output requirements:
# - Write around 900 to 1400 words, unless very little relevant evidence was retrieved.
# - Use exact NCT IDs from the retrieved context as citations.
# - Every NCT ID cited in the report body MUST appear once in the References section.
# - The References section MUST be present and must use exactly this heading:
# ## References
# - In References, list each cited NCT once as:
#   - NCT ID – Brief title
# - Never invent NCT IDs, titles, URLs, publication details, PMIDs, or journal references.
# - Write in {language}.

# {reference_prompt}
# """

#         return f"""
# Write a concise Cochrane-style evidence summary answering the research question below.

# Research question:
# "{question}"

# Retrieved ClinicalTrials.gov evidence:
# "{context}"

# Base the report only on the provided evidence. Give a direct, cautious answer and synthesize the main findings across the relevant populations, interventions, comparators, outcomes, and safety information.

# ClinicalTrials.gov records may contain either trial design information or posted results. Do not describe an outcome being measured as an observed result. Only claim benefit, harm, or safety findings when explicit results are available; otherwise state that the evidence is insufficient or that only registry design information was available.

# Write a coherent synthesis rather than a trial-by-trial list. Cite the most directly relevant records using their exact NCT IDs. Do not repeat the same NCT ID unnecessarily or produce long lists of identifiers without discussing their evidence.

# Use exactly these sections:

# ## Conclusion

# Write 3 to 5 concise paragraphs, beginning with a direct answer to the research question.

# ## References

# List each NCT ID cited in the report body once, using:
# - NCT ID – Brief title

# Do not invent NCT IDs, results, titles, publications, or other information not present in the context. Write in {language}.
# """

#         return f"""Write a concise Cochrane-style evidence summary answering the research question below.

# Research question:
# "{question}"

# Retrieved ClinicalTrials.gov evidence:
# "{context}"

# Base the report only on the provided evidence. Give a direct, cautious answer and synthesize
# the main findings across the relevant populations, interventions, comparators, outcomes, and
# safety information.

# ClinicalTrials.gov records may contain either trial design information or posted results. Do
# not describe an outcome being measured as an observed result. Only claim benefit, harm, or
# safety findings when explicit results are available; otherwise state that the evidence is
# insufficient or that only registry design information was available.

# Before writing, identify every distinct comparison the evidence supports (e.g. intervention vs.
# placebo/supportive care, intervention vs. an alternative drug class, one intervention vs.
# another within the same class). Address each distinct comparison that the evidence covers — do
# not focus only on the comparison with the richest data and omit the others.

# Write a coherent synthesis, not a trial-by-trial list: group NCT IDs together in the same
# sentence when they support the same point, rather than giving each trial its own sentence. But
# cite every directly relevant NCT ID somewhere in the report — do not limit yourself to only the
# one or two most illustrative trials per point if more relevant trials are present in the
# evidence. A synthesis sentence citing five trials at once is exactly what's wanted; a synthesis
# that quietly drops four of those five because it only wanted "the most relevant" one is not.

# If the evidence for one comparison is much thinner than for another, say so explicitly (e.g.
# "data on X's safety are insufficient") rather than omitting that comparison or overstating
# confidence in it based on a small number of trials.

# Use exactly these sections:

# ## Conclusion

# Write 3 to 5 concise paragraphs, beginning with a direct answer to the research question, then
# addressing each distinct comparison identified above in turn.

# ## References

# List each NCT ID cited in the report body once, using:
# - NCT ID – Brief title

# Do not invent NCT IDs, results, titles, publications, or other information not present in the
# context. Write in {language}.

# """

        return f""" You are a clinical evidence synthesis assistant. Write balanced and cautious summaries using only the supplied ClinicalTrials.gov evidence.

Research question:

"{question}"

Retrieved context:

<context>
{context}
</context>

Write a concise synthesis of the supplied evidence that directly answers the research question and integrates the findings across the relevant trial records.

Use three to five connected paragraphs without bullet points or internal subheadings. Begin by describing the body of directly relevant registry evidence. Then synthesise what this evidence indicates about the main comparisons, benefits, and harms addressed by the research question. Focus on the overall pattern of evidence rather than describing each trial separately. The final paragraph should provide a brief and balanced answer to the research question, reflecting the completeness and limitations of the available evidence and avoiding clinical recommendations.

Use numerical findings only when they are explicitly reported in the supplied records. Clearly distinguish posted results from planned outcomes and study-design information. When the registry evidence is limited, reflect that limitation in the strength
and wording of the conclusion.

Ignore records that are clearly unrelated to the research question. Support the main findings with grouped NCT citations, citing only the records used in the synthesis. 

Use exactly these sections:

## Summary

## References

In the References section, list each cited record once:

- NCT ID – Brief title

"""


    @staticmethod
    def auto_agent_instructions():
        return """
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific server, defined by its type and role, with each server requiring distinct instructions.
Agent
The server is determined by the field of the topic and the specific name of the server that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each server type is associated with a corresponding emoji.

examples:
task: "should I invest in apple stocks?"
response:
{
    "server": "💰 Finance Agent",
    "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "could reselling sneakers become profitable?"
response:
{
    "server":  "📈 Business Analyst Agent",
    "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
    "server":  "🌍 Travel Agent",
    "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
"""

    @staticmethod
    def generate_summary_prompt(query, data):
        """Generates the summary prompt for the given question and text.
        Args: question (str): The question to generate the summary prompt for
                text (str): The text to generate the summary prompt for
        Returns: str: The summary prompt for the given question and text
        """

        return (
            f'{data}\n Using the above text, summarize it based on the following task or query: "{query}".\n If the '
            f"query cannot be answered using the text, YOU MUST summarize the text in short.\n Include all factual "
            f"information such as numbers, stats, quotes, etc if available. "
        )

    @staticmethod
    def generate_quick_summary_prompt(query: str, context: str) -> str:
        """Generates the quick summary prompt for the given question and context.
        Args:
            query (str): The query to generate the summary for
            context (str): The search results to summarize
        Returns:
            str: The quick summary prompt
        """
        return f"""
Synthesize a comprehensive answer to the following query based ONLY on the provided search results.
Query: "{query}"

Search Results:
{context}

Instructions:
1. Provide a single, continuous narrative summary.
2. Cite your sources using numbers [1], [2], etc., corresponding to the search results.
3. If the results are insufficient to answer the query, state that clearly.
4. Focus on accuracy and relevance.
"""

    @staticmethod
    def pretty_print_docs(docs: list[Document], top_n: int | None = None) -> str:
        """Compress the list of documents into a context string"""
        return f"\n".join(f"Source: {d.metadata.get('source')}\n"
                          f"Title: {d.metadata.get('title')}\n"
                          f"Content: {d.page_content}\n"
                          for i, d in enumerate(docs)
                          if top_n is None or i < top_n)

    @staticmethod
    def join_local_web_documents(docs_context: str, web_context: str) -> str:
        """Joins local web documents with context scraped from the internet"""
        return f"Context from local documents: {docs_context}\n\nContext from web sources: {web_context}"

    ################################################################################################

    # DETAILED REPORT PROMPTS

    @staticmethod
    def generate_subtopics_prompt() -> str:
        return """
Provided the main topic:

{task}

and research data:

{data}

- Construct a list of subtopics which indicate the headers of a report document to be generated on the task.
- These are a possible list of subtopics : {subtopics}.
- There should NOT be any duplicate subtopics.
- Limit the number of subtopics to a maximum of {max_subtopics}
- Finally order the subtopics by their tasks, in a relevant and meaningful order which is presentable in a detailed report

"IMPORTANT!":
- Every subtopic MUST be relevant to the main topic and provided research data ONLY!

{format_instructions}
"""

    @staticmethod
    def generate_subtopic_report_prompt(
        current_subtopic,
        existing_headers: list,
        relevant_written_contents: list,
        main_topic: str,
        context,
        report_format: str = "apa",
        max_subsections=5,
        total_words=800,
        tone: Tone = Tone.Objective,
        language: str = "english",
    ) -> str:
        return f"""
Context:
"{context}"

Main Topic and Subtopic:
Using the latest information available, construct a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.
You must limit the number of subsections to a maximum of {max_subsections}.

Content Focus:
- The report should focus on answering the question, be well-structured, informative, in-depth, and include facts and numbers if available.
- Use markdown syntax and follow the {report_format.upper()} format.
- When presenting data, comparisons, or structured information, use markdown tables to enhance readability.

IMPORTANT:Content and Sections Uniqueness:
- This part of the instructions is crucial to ensure the content is unique and does not overlap with existing reports.
- Carefully review the existing headers and existing written contents provided below before writing any new subsections.
- Prevent any content that is already covered in the existing written contents.
- Do not use any of the existing headers as the new subsection headers.
- Do not repeat any information already covered in the existing written contents or closely related variations to avoid duplicates.
- If you have nested subsections, ensure they are unique and not covered in the existing written contents.
- Ensure that your content is entirely new and does not overlap with any information already covered in the previous subtopic reports.

"Existing Subtopic Reports":
- Existing subtopic reports and their section headers:

    {existing_headers}

- Existing written contents from previous subtopic reports:

    {relevant_written_contents}

"Structure and Formatting":
- As this sub-report will be part of a larger report, include only the main body divided into suitable subtopics without any introduction or conclusion section.

- You MUST include markdown hyperlinks to relevant source URLs wherever referenced in the report, for example:

    ### Section Header

    This is a sample text ([in-text citation](url)).

- Use H2 for the main subtopic header (##) and H3 for subsections (###).
- Use smaller Markdown headers (e.g., H2 or H3) for content structure, avoiding the largest header (H1) as it will be used for the larger report's heading.
- Organize your content into distinct sections that complement but do not overlap with existing reports.
- When adding similar or identical subsections to your report, you should clearly indicate the differences between and the new content and the existing written content from previous subtopic reports. For example:

    ### New header (similar to existing header)

    While the previous section discussed [topic A], this section will explore [topic B]."

"Date":
Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.

"IMPORTANT!":
- You MUST write the report in the following language: {language}.
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
- You MUST use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
- You MUST mention the difference between the existing content and the new content in the report if you are adding the similar or same subsections wherever necessary.
- The report should have a minimum length of {total_words} words.
- Use an {tone.value} tone throughout the report.

Do NOT add a conclusion section.
"""

    @staticmethod
    def generate_draft_titles_prompt(
        current_subtopic: str,
        main_topic: str,
        context: str,
        max_subsections: int = 5
    ) -> str:
        return f"""
"Context":
"{context}"

"Main Topic and Subtopic":
Using the latest information available, construct a draft section title headers for a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.

"Task":
1. Create a list of draft section title headers for the subtopic report.
2. Each header should be concise and relevant to the subtopic.
3. The header should't be too high level, but detailed enough to cover the main aspects of the subtopic.
4. Use markdown syntax for the headers, using H3 (###) as H1 and H2 will be used for the larger report's heading.
5. Ensure the headers cover main aspects of the subtopic.

"Structure and Formatting":
Provide the draft headers in a list format using markdown syntax, for example:

### Header 1
### Header 2
### Header 3

"IMPORTANT!":
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
- Focus solely on creating headers, not content.
"""

    @staticmethod
    def generate_report_introduction(question: str, research_summary: str = "", language: str = "english", report_format: str = "apa") -> str:
        return f"""{research_summary}\n
Using the above latest information, Prepare a detailed report introduction on the topic -- {question}.
- The introduction should be succinct, well-structured, informative with markdown syntax.
- As this introduction will be part of a larger report, do NOT include any other sections, which are generally present in a report.
- The introduction should be preceded by an H1 heading with a suitable topic for the entire report.
- You must use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
Assume that the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.
- The output must be in {language} language.
"""


    @staticmethod
    def generate_report_conclusion(query: str, report_content: str, language: str = "english", report_format: str = "apa") -> str:
        """
        Generate a concise conclusion summarizing the main findings and implications of a research report.

        Args:
            query (str): The research task or question.
            report_content (str): The content of the research report.
            language (str): The language in which the conclusion should be written.

        Returns:
            str: A concise conclusion summarizing the report's main findings and implications.
        """
        prompt = f"""
    Based on the research report below and research task, please write a concise conclusion that summarizes the main findings and their implications:

    Research task: {query}

    Research Report: {report_content}

    Your conclusion should:
    1. Recap the main points of the research
    2. Highlight the most important findings
    3. Discuss any implications or next steps
    4. Be approximately 2-3 paragraphs long

    If there is no "## Conclusion" section title written at the end of the report, please add it to the top of your conclusion.
    You must use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).

    IMPORTANT: The entire conclusion MUST be written in {language} language.

    Write the conclusion:
    """

        return prompt


class GranitePromptFamily(PromptFamily):
    """Prompts for IBM's granite models"""


    def _get_granite_class(self) -> type[PromptFamily]:
        """Get the right granite prompt family based on the version number"""
        if "3.3" in self.cfg.smart_llm:
            return Granite33PromptFamily
        if "3" in self.cfg.smart_llm:
            return Granite3PromptFamily
        # If not a known version, return the default
        return PromptFamily

    def pretty_print_docs(self, *args, **kwargs) -> str:
        return self._get_granite_class().pretty_print_docs(*args, **kwargs)

    def join_local_web_documents(self, *args, **kwargs) -> str:
        return self._get_granite_class().join_local_web_documents(*args, **kwargs)


class Granite3PromptFamily(PromptFamily):
    """Prompts for IBM's granite 3.X models (before 3.3)"""

    _DOCUMENTS_PREFIX = "<|start_of_role|>documents<|end_of_role|>\n"
    _DOCUMENTS_SUFFIX = "\n<|end_of_text|>"

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        if not docs:
            return ""
        all_documents = "\n\n".join([
            f"Document {doc.metadata.get('source', i)}\n" + \
            f"Title: {doc.metadata.get('title')}\n" + \
            doc.page_content
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """Joins local web documents using Granite's preferred format"""
        if isinstance(docs_context, str) and docs_context.startswith(cls._DOCUMENTS_PREFIX):
            docs_context = docs_context[len(cls._DOCUMENTS_PREFIX):]
        if isinstance(web_context, str) and web_context.endswith(cls._DOCUMENTS_SUFFIX):
            web_context = web_context[:-len(cls._DOCUMENTS_SUFFIX)]
        all_documents = "\n\n".join([docs_context, web_context])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])


class Granite33PromptFamily(PromptFamily):
    """Prompts for IBM's granite 3.3 models"""

    _DOCUMENT_TEMPLATE = """<|start_of_role|>document {{"document_id": "{document_id}"}}<|end_of_role|>
{document_content}<|end_of_text|>
"""

    @staticmethod
    def _get_content(doc: Document) -> str:
        doc_content = doc.page_content
        if title := doc.metadata.get("title"):
            doc_content = f"Title: {title}\n{doc_content}"
        return doc_content.strip()

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        return "\n".join([
            cls._DOCUMENT_TEMPLATE.format(
                document_id=doc.metadata.get("source", i),
                document_content=cls._get_content(doc),
            )
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """Joins local web documents using Granite's preferred format"""
        return "\n\n".join([docs_context, web_context])

## Factory ######################################################################

# This is the function signature for the various prompt generator functions
PROMPT_GENERATOR = Callable[
    [
        str,        # question
        str,        # context
        str,        # report_source
        str,        # report_format
        str | None, # tone
        int,        # total_words
        str,        # language
    ],
    str,
]

report_type_mapping = {
    ReportType.ResearchReport.value: "generate_report_prompt",
    ReportType.ResourceReport.value: "generate_resource_report_prompt",
    ReportType.OutlineReport.value: "generate_outline_report_prompt",
    ReportType.CustomReport.value: "generate_custom_report_prompt",
    ReportType.SubtopicReport.value: "generate_subtopic_report_prompt",
    ReportType.DeepResearch.value: "generate_deep_research_prompt",
}


def get_prompt_by_report_type(
    report_type: str,
    prompt_family: type[PromptFamily] | PromptFamily,
):
    prompt_by_type = getattr(prompt_family, report_type_mapping.get(report_type, ""), None)
    default_report_type = ReportType.ResearchReport.value
    if not prompt_by_type:
        warnings.warn(
            f"Invalid report type: {report_type}.\n"
            f"Please use one of the following: {', '.join([enum_value for enum_value in report_type_mapping.keys()])}\n"
            f"Using default report type: {default_report_type} prompt.",
            UserWarning,
        )
        prompt_by_type = getattr(prompt_family, report_type_mapping.get(default_report_type))
    return prompt_by_type


prompt_family_mapping = {
    PromptFamilyEnum.Default.value: PromptFamily,
    PromptFamilyEnum.Granite.value: GranitePromptFamily,
    PromptFamilyEnum.Granite3.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite31.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite32.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite33.value: Granite33PromptFamily,
}


def get_prompt_family(
    prompt_family_name: PromptFamilyEnum | str, config: Config,
) -> PromptFamily:
    """Get a prompt family by name or value."""
    if isinstance(prompt_family_name, PromptFamilyEnum):
        prompt_family_name = prompt_family_name.value
    if prompt_family := prompt_family_mapping.get(prompt_family_name):
        return prompt_family(config)
    warnings.warn(
        f"Invalid prompt family: {prompt_family_name}.\n"
        f"Please use one of the following: {', '.join([enum_value for enum_value in prompt_family_mapping.keys()])}\n"
        f"Using default prompt family: {PromptFamilyEnum.Default.value} prompt.",
        UserWarning,
    )
    return PromptFamily()
