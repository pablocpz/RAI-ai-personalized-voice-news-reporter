import asyncio
import operator
from typing_extensions import TypedDict
from typing import  Annotated, List, Optional, Literal
from pydantic import BaseModel, Field

from tavily import TavilyClient, AsyncTavilyClient

from langchain_openai import ChatOpenAI
# from lan 
from utils.configuration import Configuration
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langsmith import traceable

gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0) 
# Search

tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()
# Schema

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )



# class SearchQuery(BaseModel):
#     search_query: str = Field(None, description="Query for web search.")

# class Queries(BaseModel):
#     queries: List[SearchQuery] = Field(
#         description="List of search queries.",
#     )
class ReportStateInput(TypedDict):
    headline : str
    page_content : str

class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    # topic: str # Report topic    
    headline : str
    page_content :str
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report
    
    
from typing import List, Union

from typing import List, Union

class SearchQueriesParams(BaseModel):
    queries: List[str] = Field(
        description="A list of strings representing the search queries to be executed.",
    )
    tavily_days: List[Union[int, None]] = Field(
        description="A list of integers representing the number of days to limit search results for each query (e.g., 7 for last week), or None for no time restriction. Each value corresponds to a query in the 'queries' list.",
    )
    tavily_topic: List[str] = Field(
        description="A list of strings indicating the type of search for each query: 'news' for time-sensitive queries or 'general' for unrestricted searches. Each value corresponds to a query in the 'queries' list.",
    )

class SectionState(TypedDict):
    section: Section # Report section   
    search_queries_params: SearchQueriesParams # List of search queries
    
    page_content :str
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
# ------------------------------------------------------------
# Utility functions

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        # sources_list = search_response['results']
        sources_list = search_response.get('results', [])
        
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                # sources_list.extend(response['results'])
                sources_list.extend(response.get('results', []))
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        url = source.get('url')
        # if source['url'] not in unique_sources:
        #     unique_sources[source['url']] = source
        if url and url not in unique_sources:
            unique_sources[url] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
        {'='*60}
        Section {idx}: {section.name}
        {'='*60}
        Description:
        {section.description}
        Requires Research: 
        {section.research}

        Content:
        {section.content if section.content else '[Not yet written]'}

        """
    return formatted_str


@traceable
def tavily_search(query):
    """ Search the web using the Tavily API.
    
    Args:
        query (str): The search query to execute
        
    Returns:
        dict: Tavily search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""
     
    return tavily_client.search(query, 
                         max_results=5, 
                         include_raw_content=True)


#this tool will be called in parallel as a subgraph
@traceable
async def tavily_search_async(search_queries, tavily_topic, tavily_days=None):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        tavily_topic (str): Type of search to perform ('news' or 'general')
        tavily_days (int) or None: Number of days to look back for news articles (only used when tavily_topic='news')

    Returns:
        List[dict]: List of search results from Tavily API, one per query

    Note:
        For news searches, each result will include articles from the last `tavily_days` days.
        For general searches, the time range is unrestricted.
    """
    
    search_tasks = []
    for query in search_queries:
        if tavily_topic == "news":
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="news",
                    days=tavily_days
                )
            )
        else:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="general"
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks, return_exceptions=True)

    return search_docs

# Prompt generating the report outline
report_planner_instructions="""

You are an expert technical writer, helping to plan a report based on an input.

Your goal is to generate the outline of the sections of the report.

The report should follow this organization:

{report_organization}

The report must be strictly based on the topic inferred from the headline. Use the page content as supplementary material to guide and support the structure and detail of the sections. 

You have the following inputs to guide the structuring of the report:

- **News item Headline**: {headline} \n\n
- **Page Content**: {page_content} \n\n

Infer the central topic of the report strictly from the headline. Base the report structure on this topic while ensuring that the sections are relevant to and supported by the provided page content. Avoid performing additional web research at this stage unless explicitly specified.

Now, generate the sections of the report. Each section should have the following fields:

- **Name**: Name for this section of the report.
- **Description**: Brief overview of the main topics and concepts to be covered in this section.
- **Research**: Whether to perform web research for this section of the report (set to "False" if the content is sufficiently covered in the page, set it to True otherwise, if you think it will be necessary).
- **Content**: The content of the section, which you will leave blank for now.

\n\n Ensure the structure remains aligned with the topic derived from the headline and that the page content informs the section descriptions. Avoid introducing unrelated themes or concepts.

Consider which sections require web research. For example, introduction and conclusion will not require research because they will distill information from other parts of the report.
"""

# Query writer instructions
query_writer_instructions="""

Your goal is to generate targeted web search queries that will gather comprehensive and precise information for writing a technical report section.

### Inputs for Query Generation:
- **Section Topic**: {section_topic}  
- **Original News Item Content**:  
  The news item content will be provided in a separate message below. Please refer to the provided **news_page_content** that will follow this system message for further context.  

Use the section topic as the primary focus, but refer to the original news item content to identify:  
1. **Information already covered**: Avoid redundant queries that overlap with content already present.  
2. **Gaps in coverage**: Focus queries on missing or supplementary aspects needed to enhance the section.

### Guidelines for Query Construction:

1. **Coverage**:  
   Queries should cover diverse and essential aspects of the section topic, such as:  
   - Core features and attributes.  
   - Applications and real-world examples.  
   - Historical and contextual background (if applicable).  
   - Technical challenges or advancements.  
   - Relevant comparisons or differentiators.

2. **Specificity**:  
   - Use technical and domain-specific terminology.  
   - Avoid overly generic phrasing to minimize irrelevant results.  

3. **Relevance**:  
   - Consider time sensitivity:  
     - If the topic is tied to current events (e.g., "latest updates on X" or "recent advancements"), set `tavily_topic="news"` and enable `tavily_days=7`.  
     - For historical or enduring topics, use `tavily_topic="general"` and do not limit by date (`tavily_days=None`).  

4. **Authority**:  
   - Target authoritative sources such as:  
     - Official documentation.  
     - Technical blogs by experts.  
     - Peer-reviewed academic papers.  
     - Reputable industry publications.  

5. **Freshness**:  
   - Include year markers (e.g., "2024") when recent developments are crucial to the query.  

6. **User Expertise**:  
   Tailor queries to accommodate the user's background:  
   - Avoid unnecessary jargon in topics where the user has basic knowledge (e.g., politics or economy).  
   - Include explanatory queries to provide foundational understanding of advanced concepts, such as economic terms ("What is inflation and how does it work?").  
   - Assume familiarity with major current wars (e.g., Gaza-Israel, Ukraine-Russia).  
   - Emphasize the root causes and historical context of issues to align with the user's interest in understanding deeper insights.  
   - Use precise and technical terms for AI and technology topics, as the user is knowledgeable in these areas.  

### Tavily-Specific Web Search Parameters:
When formulating queries, set the Tavily parameters as follows:
- **tavily_topic**:  
   - Use `"news"` if the section deals with time-sensitive or current events.  
   - Use `"general"` for technical, historical, or non-time-sensitive topics.  
- **tavily_days**:  
   - Set `7` only if `tavily_topic="news"`.  
   - Set `None` if `tavily_topic="general"`.  

### Output:
Generate {number_of_queries} high-quality queries that:
- Reflect the section topic and its unique requirements.  
- Account for the user's background knowledge to ensure relevance and accessibility.  
- Take into account the Tavily parameters and their appropriate use cases.  
- Avoid redundancy with the provided news item content.  

Be precise and intentional with each query to ensure efficient and relevant search results.

---

**Important Note**:  
The **news_page_content** will be sent in a separate message. Please refer to that content accordingly to avoid redundancy and ensure queries are precise.

"""


# Section writer instructions
section_writer_instructions = """
You are an expert technical writer crafting one section of a technical report.

### Inputs for Writing the Section:
- **Section Topic**: {section_topic}
- **News Item Content**:  
  The content of the news item will be provided **in a separate message**. This content should be used to complement the retrieved search data and serve as the starting point for the section. 
- **User Background**:  
  The user has the following background and preferences:
  - Barely knows about politics (basic concepts like left-wing, right-wing). Explain political concepts in simple and concise terms.
  - Knows about the current wars (e.g., Gaza-Israel, Ukraine-Russia).
  - Has high school-level knowledge in science and is more focused on technology and AI.
  - Does not know much about economics (e.g., inflation). Ensure economic concepts are explained simply.
  - Prefers understanding the root of problems and gaining historical insights. Strive to break down complex topics and explain their historical context.

### Guidelines for Writing:

1. **Technical Accuracy**:
   - Include specific version numbers.
   - Reference concrete metrics/benchmarks.
   - Cite official documentation.
   - Use technical terminology precisely.

2. **Length and Style**:
   - Strict 150-200 word limit.
   - No marketing language.
   - Technical focus.
   - Write in simple, clear language.
   - Start with your most important insight in **bold**.
   - Use short paragraphs (2-3 sentences max).

3. **Structure**:
   - Use `##` for section title (Markdown format).
   - Only use ONE structural element IF it helps clarify your point:
     * Either a focused table comparing 2-3 key items (using Markdown table syntax).
     * Or a short list (3-5 items) using proper Markdown list syntax:
       - Use `*` or `-` for unordered lists.
       - Use `1.` for ordered lists.
       - Ensure proper indentation and spacing.
   - End with `### Sources` that references the below source material formatted as:
     * List each source with title, date, and URL.
     * Format: `- Title : URL`.

4. **Writing Approach**:
   - Include at least one specific example or case study.
   - Use concrete details over general statements.
   - Make every word count.
   - No preamble prior to creating the section content.
   - Focus on your single most important point.

5. **Use this source material to help write the section**:
   {context}  
   Refer to the **news_page_content** (which will be provided in a separate message) for further insights and use it to form the foundation for the section.  
   Consider the user's background and preferences to adjust the tone and depth of explanation.

6. **Quality Checks**:
   - Exactly 150-200 words (excluding title and sources).
   - Careful use of only ONE structural element (table or list) and only if it helps clarify your point.
   - One specific example / case study.
   - Starts with bold insight.
   - No preamble prior to creating the section content.
   - Sources cited at end.
"""

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

Section to write: 
{section_topic}

Available report content:
{context}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response"""

import logging
# Graph nodes

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """ Generate the report plan """

    # Inputs
    # topic = state["topic"]
    
    headline = state["headline"]
    page_content = state["page_content"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    # tavily_topic = configurable.tavily_topic
    # tavily_days = configurable.tavily_days

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Generate search query
    structured_llm = gpt_4o.with_structured_output(Sections)

    # Format system instructions

    system_instructions_sections = report_planner_instructions.format(report_organization=report_structure, headline=headline, page_content=page_content)
    
    
    

    # Generate sections 
    structured_llm = gpt_4o.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections)]+[HumanMessage(content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields.")])

    
    
    logging.debug("Sections Outline Written!")
    
    return {"sections": report_sections.sections    }



def generate_queries(state: SectionState, config: RunnableConfig):
    """ Generate search queries for a report section, and set tavily_topic and tavily_days"""

    # Get state 
    section = state["section"]
    page_content = state["page_content"] if "page_content" in state.keys() else ""

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries and tavily params with the custom pydantic model
    structured_llm = gpt_4o.with_structured_output(SearchQueriesParams)

    # Format system instructions
    system_instructions = query_writer_instructions.format(section_topic=section.description, number_of_queries=number_of_queries,
                                                           )

    # Generate queries  
    search_params = structured_llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content=f"Generate search queries on the provided topic and set the value of the params tavily_topic and tavily_days. The reference news page content is {page_content}")])

    queries = search_params.queries  # Assuming 'queries' is the result you expect from the LLM response
    tavily_topic = search_params.tavily_topic  # LLM response for tavily_topic
    tavily_days = search_params.tavily_days
    
    search_queries_params = SearchQueriesParams(
        queries=queries,
        tavily_topic=tavily_topic,
        tavily_days=tavily_days
    )
    
    logging.debug("Search queries params node executed sucessfully!")
    
    return {"search_queries_params": search_queries_params}


async def search_web(state: SectionState):
    """ Search the web for each query, then return a list of raw sources and a formatted string of sources."""
    
    # Get search_queries_params from state (which includes lists of queries, tavily_topic, and tavily_days)
    search_queries_params = state["search_queries_params"]
    queries = search_queries_params.queries  # List of queries
    tavily_topics = search_queries_params.tavily_topic  # List of topics ('news' or 'general')
    tavily_days = search_queries_params.tavily_days  # List of days for limiting search (or None)

    # Check if lengths of queries, tavily_topics, and tavily_days are equal
    if not (len(queries) == len(tavily_topics) == len(tavily_days)):
        raise ValueError("The lengths of queries, tavily_topics, and tavily_days must be equal.")
    
    # Web search using async function with corresponding parameters
    search_docs = []
    for query, topic, days in zip(queries, tavily_topics, tavily_days):
        search_doc = await tavily_search_async([query], topic, days)
        search_docs.extend(search_doc)  # Add results to the overall list

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000, include_raw_content=True)

    
    logging.debug("Web search node done sucessfully!")
    return {"source_str": source_str}



def  write_section(state: SectionState):
    """ Write a section of the report """

    # Get state 
    section = state["section"]
    source_str = state["source_str"]
    page_content = state["page_content"]

    # Format system instructions
    system_instructions = section_writer_instructions.format(section_title=section.name, section_topic=section.description, context=source_str)

    # Generate section  
    section_content = gpt_4o.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content=f"Generate a report section based on the provided sources, and the news item page content: \n\n {page_content}")])
    
    # Write content to the section object  
    section.content = section_content.content
    
    
    logging.debug(f"Section {section.name} (with research={section.research}), written sucessfully!")
    
    # Write the updated section to completed sections
    return {"completed_sections": [section]}



# Add nodes and edges 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")
section_builder.add_edge("write_section", END)


def initiate_section_writing(state: ReportState):
    """ This is the "map" step when we kick off web research for some sections of the report """    
    # Debug: Log the sections and their research flag
    for section in state["sections"]:
        logging.debug(f"Section: {section}, Research flag: {section.research}")

    tasks = [
        Send("build_section_with_web_research", {"section": s, "page_content" : state["page_content"]}) 
            #this is the "sender" node, along with it's inputs {} for the state that this subgraph expects
        for s in state["sections"] 
        if s.research
    ]
    
    # Log each Send task
    for task in tasks:
        logging.debug(f"Initiating task: {task}")
        
    return tasks



def write_final_sections(state: SectionState):
    """ Write final sections of the report, which do not require web search and use the completed sections as context """

    # Get state 
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(section_title=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    section_content = gpt_4o.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


def gather_completed_sections(state: ReportState):
    """ Gather completed sections from research and format them as context for writing the final sections """    

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def initiate_final_section_writing(state: ReportState):
    """ Write any final sections using the Send API to parallelize the process """    

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]
    
    
def compile_final_report(state: ReportState):
    """ Compile the final report """    

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}


# Add nodes and edges 
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)


builder.add_edge(START, "generate_report_plan")
builder.add_conditional_edges("generate_report_plan", initiate_section_writing, ["build_section_with_web_research"])
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()