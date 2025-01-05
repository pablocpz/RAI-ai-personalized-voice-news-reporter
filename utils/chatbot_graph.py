from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langsmith import traceable
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import json
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from tavily import TavilyClient, AsyncTavilyClient
# tavily_async_client = AsyncTavilyClient()
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Union
tavily_client = TavilyClient()


### State
class SearchQueriesParams(BaseModel):
    n_queries : int=Field(description="Number of queries to generate")
    
    queries: List[str] = Field(
        description="A list of strings representing the search queries to be executed.",
    )
    tavily_days: List[Union[int, None]] = Field(
        description="A list of integers representing the number of days to limit search results for each query (e.g., 7 for last week), or None for no time restriction. Each value corresponds to a query in the 'queries' list.",
    )
    tavily_topic: List[str] = Field(
        description="A list of strings indicating the type of search for each query: 'news' for time-sensitive queries or 'general' for unrestricted searches. Each value corresponds to a query in the 'queries' list.",
    )


#we'll reference to this object very often to add new docs...etc
    
class GraphState(TypedDict):
    news_summary : str
    question : str
    question_type : Union[str, str]
    generation : str
    web_search : Union[str, str] 
    search_queries_params : SearchQueriesParams
    documents : str #the concatenated list of documents text content (or search results)
    decission : str
    feedback : Union[None, str]
    iterations:int


# Check if the directory and files are readable
directory = './rag_docs'
# print(os.access(directory, os.R_OK))  # Checks if the directory is readable
os.chmod('./rag_docs', 0o755)


markdown_folder_path = "./rag_docs"  # Set the path to your folder
documents = []

# Iterate over all .md files in the directory
for file in os.listdir(markdown_folder_path):
    if file.endswith('.md'):
        markdown_path = os.path.join(markdown_folder_path, file)                     #=fast
        loader = UnstructuredMarkdownLoader(markdown_path, mode="single", strategy="precise")
        documents.extend(loader.load())  # Add loaded documents to the list

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=100, add_start_index=True #starting char pos.
    #size of characters for each chunk
    #2nd param: will let us have a little portion of the prev. chunk
    # so in case the key info is in that chunk, we can have a way to get those prev chars if needed 
    
    #
)
all_splits = text_splitter.split_documents(documents)

#the so-called chroma database
local_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
persist_dir = "./chroma_db"
os.makedirs(persist_dir, exist_ok=True)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings,
                                    persist_directory=persist_dir)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3},
                                    )


llm = ChatOpenAI(model="gpt-4o", temperature=0) 

llm_json = ChatOpenAI(
        model="gpt-4o",
        temperature = 0,
        model_kwargs={"response_format": {"type": "json_object"}})


#---------------------
#prompts



# docs_answer_generation_instructions = """
#  You are an assistant for question-answering tasks inside a phone call conversation.
 
 
 
#     Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
#     Keep the answer concise optimized to reduce latency, in a spoken language style.
    
#     Question: {question} 
#     Context: {context} 
    
#     You must also keep in mind that you're answering questions related with the news of the week, so you must always respond properly connecting the answer with the spoken topics, for which i'll attach below as a brief summary.
# """

#TODO: we are trying to address the case where it gets an input question which is missleading
docs_answer_generation_instructions = """
You are an assistant for question-answering tasks inside a phone call conversation.

Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and optimized to reduce latency, in a spoken language style.

If provided, there will be a separate message with feedback about what to improve from the previous generation attempt. Use this feedback to enhance your response while maintaining grounding in the provided context.

Additionally, evaluate whether the provided context contradicts, corrects, or invalidates the user's question. If you are 100% certain that the question is misleading, incorrect, or based on faulty assumptions, use the context to explain why the question is problematic and provide the correct information. In such case, justify your answer with explicit references to the context. 

You must also keep in mind that you're answering questions related to either the news of the week, current issues, or historical questions, so always respond appropriately by connecting your answer to the spoken topics, which I'll attach below as a brief summary.

Question: {question}
Context: {context}

Remember: 
- Only correct or challenge the user's question when the context provides indisputable evidence to do so. Otherwise, answer normally.
- If feedback was provided, ensure your response addresses all the improvement points while staying faithful to the context.
"""

# query_writer_instructions = """

# Query Writer Instructions:
# ---
# Given the user's background and the question provided below, generate an appropriate number (n) of relevant queries to search on the internet in order to gather sufficient information to answer the question.

# If provided, there will be a separate message with feedback about why previous queries were not useful. Use this feedback to avoid similar issues and generate more effective queries.

# **Current Date**: {current_date}

# **Important Note**: A separate message includes a list of previously searched queries (`searched_queries`). Consider these queries and evaluate why they may not have been useful (e.g., irrelevant results, overly broad/narrow focus, or failure to address the question fully). Use this evaluation and any provided feedback to refine, reformulate, or supplement the queries to ensure better results. If no `searched_queries` are provided, proceed without them.

# **Datetime Marker Awareness**:  
# When generating queries, use datetime markers only if appropriate to the query context:  

# 1. **Time-Sensitive or Current Events (`tavily_topic="news" and tavily_days=7`)**:  
#    - **Do not include any explicit datetime markers** (e.g., "2025" or "January 2025") in the search query, as `tavily_days=7` already ensures the query focuses on recent results.  
#    - If the user's question explicitly includes time hints (e.g., "yesterday," "last week"), calculate the corresponding date relative to **Current Date** (e.g., "yesterday" = `{current_date} - 1 day`) and use this date in the query.  
#    - Otherwise, avoid adding datetime markers for time-sensitive queries, as they are unnecessary and may limit search results.  

# 2. **General or Historical Topics (`tavily_topic="general" and tavily_days=None`)**:  
#    - Include datetime markers only when they clarify the query or provide precision (e.g., "economic crisis Spain 2008").  
#    - Avoid unnecessary datetime markers for well-known historical events (e.g., "Battle of Lepanto"), as the event is self-contained in history.  

# 3. **Redundancy**: Ensure datetime markers are added only if needed for precision based on the query context and `tavily` parameters.  

# Failure to follow these guidelines may result in ineffective or overly narrow queries.

# ### User Context:
# - **User's Knowledge**: The user knows basic political concepts (e.g., left-wing, right-wing) but may require simple, concise explanations of political contexts. They are familiar with global conflicts (e.g., Gaza-Israel, Ukraine-Russia) but are not an expert in political theory.
# - **User's Interest**: The user is interested in technology and AI and has high school-level science knowledge, so technical concepts can be explained with that in mind.
# - **User's Gaps in Knowledge**: The user has limited understanding of economic concepts (e.g., inflation, stock markets), so explanations related to economics should be kept simple and direct.
# - **Contextual Relevance**: Ensure that the queries are tailored to cover both current events (as presented in news) and any necessary historical background. The queries should also bridge any gaps in the user's knowledge without overwhelming them with unnecessary detail.

# ### Query Complexity and Necessity:
# - If the question is straightforward (e.g., "What is the capital of Spain?"), generate a single query.
# - For more complex or multi-faceted questions, generate multiple queries to ensure all relevant aspects are covered without redundancy.
# - If feedback was provided, adjust query specificity and coverage accordingly.

# ### Tavily-Specific Web Search Parameters:
# For each query, set the Tavily parameters as follows:
# - **tavily_topic**:  
#    - Use `"news"` if the query is related to time-sensitive or current events.  
#    - Use `"general"` for technical, historical, or non-time-sensitive topics.  
# - **tavily_days**:  
#    - Set `7` only if `tavily_topic="news"`.  
#    - Set `None` if `tavily_topic="general"`.

# ---

# User question: {question}


# """

query_writer_instructions = """
Given the user's background and the question provided below, generate an appropriate number (n) of relevant queries to search on the internet in order to gather sufficient information to answer the question.

If provided, there will be a separate message with feedback about why previous queries were not useful. Use this feedback to avoid similar issues and generate more effective queries.

**Current Date**: {current_date}

**Important Note**: A separate message includes a list of previously searched queries (`searched_queries`). Consider these queries and evaluate why they may not have been useful (e.g., irrelevant results, overly broad/narrow focus, or failure to address the question fully). Use this evaluation and any provided feedback to refine, reformulate, or supplement the queries to ensure better results. If no `searched_queries` are provided, proceed without them.

**Datetime Marker Awareness**:  
When generating queries, use datetime markers only if appropriate to the query context:

1. **Time-Sensitive or Current Events (`tavily_topic="news" and tavily_days=7`)**:  
   - **Do not include any explicit datetime markers** (e.g., "2025" or "January 2025") in the search query, as `tavily_days=7` already ensures the query focuses on recent results.  
   - If the user's question explicitly includes time hints (e.g., "yesterday," "last week"), calculate the corresponding date relative to **Current Date** (e.g., "yesterday" = `{current_date} - 1 day`) and use this date in the query.  
   - Otherwise, avoid adding datetime markers for time-sensitive queries, as they are unnecessary and may limit search results.  

2. **General or Historical Topics (`tavily_topic="general" and tavily_days=None`)**:  
   - Include datetime markers only when they clarify the query or provide precision (e.g., "economic crisis Spain 2008").  
   - Avoid unnecessary datetime markers for well-known historical events (e.g., "Battle of Lepanto"), as the event is self-contained in history.  

3. **Redundancy**: Ensure datetime markers are added only if needed for precision based on the query context and `tavily` parameters.  

Failure to follow these guidelines may result in ineffective or overly narrow queries.

### User Context:
- **User's Knowledge**: The user knows basic political concepts (e.g., left-wing, right-wing) but may require simple, concise explanations of political contexts. They are familiar with global conflicts (e.g., Gaza-Israel, Ukraine-Russia) but are not an expert in political theory.
- **User's Interest**: The user is interested in technology and AI and has high school-level science knowledge, so technical concepts can be explained with that in mind.
- **User's Gaps in Knowledge**: The user has limited understanding of economic concepts (e.g., inflation, stock markets), so explanations related to economics should be kept simple and direct.
- **Contextual Relevance**: Ensure that the queries are tailored to cover both current events (as presented in news) and any necessary historical background. The queries should also bridge any gaps in the user's knowledge without overwhelming them with unnecessary detail.

### Query Complexity and Necessity:
- If the question is straightforward (e.g., "What is the capital of Spain?"), generate a single query.
- For more complex or multi-faceted questions, generate multiple queries to ensure all relevant aspects are covered without redundancy.
- If feedback was provided, adjust query specificity and coverage accordingly.

### Tavily-Specific Web Search Parameters:
For each query, set the Tavily parameters as follows:
- **tavily_topic**:  
   - Use `"news"` if the query is related to time-sensitive or current events.  
   - Use `"general"` for technical, historical, or non-time-sensitive topics.  
- **tavily_days**:  
   - Set `7` only if `tavily_topic="news"`.  
   - Set `None` if `tavily_topic="general"`.

---

**User question**: {question}
"""

docs_grader_instructions = """

    You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.

    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question}
"""

# generation_grader_instructions = """
# You are a grader assessing whether an answer is grounded in or supported by a set of facts. Additionally, evaluate whether the answer correctly identifies and addresses any misleading or incorrect assumptions in the user's question, based on the provided facts. In such case, set to "yes" in the score key of the output JSON.

# Provide your assessment as a JSON with:
# - A 'score' key with value 'yes' or 'no'
# - If score is 'no', include a 'feedback' key with specific guidance that will be passed to the answer generation system in a separate message. The feedback should focus on:
#   * Which claims need better support from the documents
#   * What information was misinterpreted or omitted
#   * How to better structure the answer for a phone conversation
#   * What aspects need to be explained more concisely
#   * How to better integrate the weekly news context

# The feedback will be used by the docs_answer_generation system to create a new answer using the same documents.

# Here are the facts:
# \n ------- \n
# {documents}
# \n ------- \n

# Here is the generated answer:
# \n ------- \n
# {generation}
# \n ------- \n

# Respond only with the JSON object, no explanation."""

generation_grader_instructions = """

You are a grader assessing whether an answer is grounded in or supported by a set of facts. Additionally, evaluate whether the answer correctly identifies and explicitly addresses any misleading or incorrect assumptions in the user's question, based on the provided facts.

If the answer correctly identifies and addresses such misleading or incorrect assumptions, assign a `"yes"` to the `score` key, even if the user's question contains inaccuracies.

For example:
- Question: "Who was Fernando Hitler?"
  * Incorrect Assumption: The name "Fernando Hitler" is historically inaccurate.
  * Correct Response: Clarifies that no person named Fernando Hitler exists in historical records and offers accurate information about Adolf Hitler if relevant.
  * Outcome: Score "yes" because the response correctly addresses the misleading premise.

Your task:
1. Determine whether the answer is factually grounded and addresses misleading or incorrect assumptions.
2. If the answer successfully corrects such assumptions and provides a grounded response, score `"yes"`.
3. If the answer fails, provide feedback to improve it.

Provide your assessment as a JSON with:
- A 'score' key with value 'yes' or 'no'
- If the score is 'no', include a 'feedback' string key with specific guidance on:
  * Identifying and addressing misleading assumptions.
  * Correcting factual errors or omissions.
  * Enhancing clarity and conciseness.
  * Aligning with weekly news context if applicable.

Here are the facts:
\n ------- \n
{documents}
\n ------- \n

Here is the generated answer:
\n ------- \n
{generation}
\n ------- \n

Respond only with the JSON object, no explanation.

"""

# answer_grader_instructions = """

# You are a grader assessing whether an answer is useful and effectively resolves the user's question. Additionally, evaluate whether the answer appropriately challenges and corrects any misleading or incorrect assumptions in the question, if the context justifies such corrections.

# If the answer correctly identifies and addresses such misleading assumptions, assign a `"yes"` to the `score` key, even if the user's question contains inaccuracies.

# For example:
# - Question: "Who was Fernando Hitler?"
#   * Incorrect Assumption: The name "Fernando Hitler" is historically inaccurate.
#   * Correct Response: Clarifies that no individual named Fernando Hitler exists in historical records, redirecting the user to relevant facts about Adolf Hitler if applicable.
#   * Outcome: Score "yes" because the response correctly addresses the misleading premise.

# Your task:
# 1. Evaluate whether the answer resolves the user's question and identifies misleading assumptions when applicable.
# 2. If the answer successfully corrects these issues and is useful, score `"yes"`.
# 3. If the answer fails, provide feedback on how it can improve.

# Provide your assessment as a JSON with:
# - A 'score' key with value 'yes' or 'no'
# - If the score is 'no', include a 'feedback' key focusing on:
#   * Filling information gaps.
#   * Suggesting better search terms or document usage.
#   * Addressing specific misleading assumptions in the question.
#   * Aligning with the appropriate level of technical detail for the user.

# Here is the generated answer:
# \n ------- \n
# {generation}
# \n ------- \n

# Here is the original user question:
# \n ------- \n
# {question}
# \n ------- \n

# Respond only with the JSON object, no explanation.

# # """

answer_grader_instructions = """
You are a grader assessing whether an answer is useful and effectively resolves the user's question. Additionally, evaluate whether the answer appropriately challenges and corrects any misleading or incorrect assumptions in the question, if the context justifies such corrections.

If the answer correctly identifies and addresses such misleading assumptions, assign a `"yes"` to the `score` key, even if the user's question contains inaccuracies.

For example:
- Question: "Who was Fernando Hitler?"
  * Incorrect Assumption: The name "Fernando Hitler" is historically inaccurate.
  * Correct Response: Clarifies that no individual named Fernando Hitler exists in historical records, redirecting the user to relevant facts about Adolf Hitler if applicable.
  * Outcome: Score "yes" because the response correctly addresses the misleading premise.

Your task:
1. Evaluate whether the answer resolves the user's question and identifies misleading assumptions when applicable.
2. If the answer successfully corrects these issues and is useful, score `"yes"`.
3. If the answer fails, provide feedback on how it can improve.

### Specific Evaluation Criteria for Query Generation:
If the user's question pertains to **time-sensitive or current events**, and:
1. The generated queries incorrectly use explicit datetime markers (e.g., "January 2025") despite `tavily_days=7` and `tavily_topic="news"`, assign `"no"` to the `score` key.
2. If the user's question does not include any explicit temporal hints (e.g., "yesterday"), and the queries still include unnecessary datetime markers, this is a misuse and warrants `"no"`.
3. If the answer does not appropriately tailor queries to the user's question (e.g., redundant or overly broad), assign `"no"`.

When assigning a `"no"` score, provide detailed and actionable feedback under the `feedback` key to ensure the query generation improves. Your feedback should:
- Highlight the improper use of datetime markers if applicable.
- Suggest excluding datetime markers when `tavily_days=7` and the question lacks explicit temporal hints.
- Emphasize aligning query parameters (e.g., `tavily_topic` and `tavily_days`) with the user's question context.
- Recommend clearer or more precise query formulations to address the user's needs.

### Additional Context:
- **Already Searched Queries**: You will receive the list of already searched queries as a separate message below. Use this information to evaluate whether the generated queries address gaps, improve upon previous attempts, and avoid redundant or ineffective formulations.

Provide your assessment as a JSON with:
- A `score` key with value `"yes"` or `"no"`.
- If the score is `"no"`, include a `feedback` key focusing on:
  * Addressing improper datetime marker usage.
  * Improving query clarity and relevance.
  * Adhering to `tavily` parameter requirements.
  * Filling gaps in information and tailoring to the user’s question.

Here is the generated answer:
\n ------- \n
{generation}
\n ------- \n

Here is the original user question:
\n ------- \n
{question}
\n ------- \n

Here are the already searched queries:
\n ------- \n
{searched_queries}
\n ------- \n

Respond only with the JSON object, no explanation.

# """


# answer_grader_instructions = """
# You are a grader assessing whether an answer is useful to resolve a question. Additionally, determine whether the answer appropriately challenges and corrects any misleading or incorrect assumptions in the question, if the provided context justifies such a correction.

# Provide your assessment as a JSON with:
# - A 'score' key with value 'yes' or 'no'
# - If score is 'no', include a 'feedback' key that will be passed to the query generation system in a separate message. The feedback should focus on:
#   * What specific information gaps need to be filled
#   * What search terms might yield better results
#   * What time periods should be considered (for Tavily parameters)
#   * What aspects of the topic need more targeted queries
#   * What level of technical detail is appropriate given the user profile

# The feedback will be used by the query_writer system to generate new, more effective search queries.

# Here is the generated answer:
# \n ------- \n
# {generation}
# \n ------- \n

# Here is the original user question:
# \n ------- \n
# {question}
# \n ------- \n

# Respond only with the JSON object, no explanation."""

#--------------- nodes

@traceable
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    # print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    retrieved_docs = retriever.invoke(question)
    
    context = ' '.join([doc.page_content for doc in retrieved_docs])
    
    return {"documents": context, "question": question}
#

@traceable
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    # print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    context = state["documents"]
    
    
    system_instructions = docs_grader_instructions.format(question=question, document=context)
    messages = [
        {"role": "system", "content": system_instructions},
        # {"role": "user", "content": f": {}"}
    ]   
    grade = json.loads(llm_json.invoke(messages).content)["score"]
    # print(grade)
    
    if grade.lower() == "yes":
        # print("---GRADE: DOCUMENT RELEVANT---")
        return {"documents": context, "question": question, "web_search": "No"}
    else:
        # print("---GRADE: RETRIEVED DOCUMENTS NOT RELEVANT, RUNNING WEB SEARCH INSTEAD---")
        return {"documents": context, "question": question, "web_search": "Yes"}


@traceable
def handle_trivial_question(state):
    """
    Determines whether the question is trivial or irrelevant to run RAG or websearch.
    Uses an LLM to classify the question as "is_trivial" or "not_trivial".

    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updated state with the classification ("is_trivial" or "not_trivial").
    """
    
    question = state["question"]
    
    # LLM prompt to classify the question as trivial or not
    trivial_classification_instructions = """
    You are an AI chatbot. Classify the user's question as either "is_trivial" or "not_trivial". 
    The question is considered "trivial" if it is a greeting, a simple personal question (e.g., "What's your name?"), 
    , incomplete or missleading questions, or if it doesn't require to search for detailed information to answer.
    If the question is a valid request for information or involves specific queries that require processing or websearch, 
    classify it as "not_trivial".
    
    Example trivial questions: 
    - "Hello"
    - "What's your name?"
    - "How are you?"
    
    Example non-trivial questions:
    - "Tell me the latest news on AI"
    - "What is the capital of France?"
    - "How does a neural network work?"
    
    Question: {question}
    Please respond with either "is_trivial" or "not_trivial", no preambles or explanations
    """

    # Create the system message with the classification instructions and the question
    system_instructions = trivial_classification_instructions.format(question=question)
    
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": f"Question: {question}"}
    ]    

    # Get the classification from the LLM
    decission = llm.invoke(messages).content.lower()
    
    if decission == "is_trivial":
        return {"question": question, "question_type": "is_trivial", "iterations":0}
    
    elif decission == "not_trivial":
        return {"question": question, "question_type": "not_trivial", "iterations":0}
    
    
@traceable
def answer_trivial_question(state):
    """
    Handle trivial or irrelevant questions by using an LLM to generate a response.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Updated state with a response to the trivial question.
    """
    question = state["question"]

    # Trivial question generation instructions
    # trivial_answer_generation_instructions = """
    # You are an AI chatbot. 
    # """

    # # Send the question and instructions to the LLM to generate a response
    # system_instructions = trivial_answer_generation_instructions + f"User's question: {question}"

    # messages = [
    #     {"role": "system", "content": system_instructions},
    #     {"role": "user", "content": f"Question: {question}"}
    # ]    
    
    messages = [
        {"role": "system", "content": "You are an AI chatbot answering a user's question."},
        {"role": "user", "content": f"Question: {question}"}
    ]    

    # Call the LLM to generate the response
    response = llm.invoke(messages).content
    
    # Return the state with the generated response for trivial question
    return {"question": question, "generation": response}
#                                               #we put the generation key (final answer) in the state
    

@traceable
def answer_with_docs(state):
    """
    Generate answer either using RAG on retrieved documents or web search results

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    # print("---GENERATE---")
    question = state["question"]
    context = state["documents"]
    
    feedback = state.get("feedback", None)
    
    
    system_instructions = docs_answer_generation_instructions.format(question=question, context=context)
    
    messages = [
        {"role": "system", "content": system_instructions},
        #todo: check if this additional param is necessary
        {"role": "user", "content": f": Here is the news summary shown to the user before: {state['news_summary']}"},
        {"role":"user", "content":f"Here is some feedback (if applicable): {feedback}"}
    ]    
    answer = llm.invoke(messages).content
     
    return {"documents": context, "question": question, "generation": answer}
#

@traceable
def generate_queries(state):
    """ Generate search queries for a report section, and set tavily_topic and tavily_days"""

    
    feedback = state.get("feedback", None)
    
    # Generate queries and tavily params with the custom pydantic model
    structured_llm = llm.with_structured_output(SearchQueriesParams)

    #already searched queries    
    searched_queries = state.get("search_queries_params", None) #looking for already searched queries
    
    
    current_datetime = datetime.now()

# Format it as a string (optional, e.g., 'YYYY-MM-DD HH:MM:SS')
    current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    # Format system instructions                   #instead of number_of_queries=3, now it also decides such number
    system_instructions = query_writer_instructions.format(question=state["question"],
                                                           current_date=current_datetime_str,
                                                        )

    print(f"already searched queries (if applicable): {searched_queries}")
    
    messages = [{"role": "system", "content": system_instructions},
                {"role":"user", "content":f"feedback (if applicable): {feedback}"}]
    
    if searched_queries is not None:
        messages.append({"role": "user", "content": f"already searched queries (if applicable): {searched_queries}"})


    search_params = structured_llm.invoke(messages)
    
    n_queries = search_params.n_queries
    queries = search_params.queries  # Assuming 'queries' is the result you expect from the LLM response
    tavily_topic = search_params.tavily_topic  # LLM response for tavily_topic
    tavily_days = search_params.tavily_days
    
    search_queries_params = SearchQueriesParams(
        n_queries=n_queries,
        queries=queries,
        tavily_topic=tavily_topic,
        tavily_days=tavily_days
    )
    
    
    return {"search_queries_params": search_queries_params}


#TODO: we need to make all the graph either async or sync, not mixed , see https://github.com/langchain-ai/langgraph/issues/2928#issuecomment-2569915286

#TODO NOTE: we have selected to get n=3 results per query call (maybe we should try setting it to 1 to reduce latency)
#or keeping such amount to ensure it retrieves trustworthy sources

def tavily_search_sync(search_queries:List[str], tavily_topic:str, tavily_days=None):
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
    search_docs = []
    for query in search_queries:
        if tavily_topic == "news":
            result = tavily_client.search(  # Assuming `tavily_client` is the synchronous equivalent of `tavily_async_client`
                query,
                max_results=1, #1 #todo: ---> DECIDE THE NUMBER OF RESULTS
                include_raw_content=True,
                topic="news",
                days=tavily_days
            )
        else:
            result = tavily_client.search(
                query,
                max_results=1,#1 #todo: ---> DECIDE THE NUMBER OF RESULTS
                include_raw_content=True,
                topic="general"
            )
        search_docs.append(result)

    return search_docs


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


def search_web(state):
    """ Web search based based on the question."""
    
    # Get search_queries_params from state (which includes lists of queries, tavily_topic, and tavily_days)
     
    search_queries_params = state["search_queries_params"]
    n_queries = search_queries_params.n_queries  # Number of queries to generate
    queries = search_queries_params.queries  # List of queries
    tavily_topics = search_queries_params.tavily_topic  # List of topics ('news' or 'general')
    tavily_days = search_queries_params.tavily_days  # List of days for limiting search (or None)

    print(f"gonna generate {n_queries} queries")
    # Check if lengths of queries, tavily_topics, and tavily_days are equal
    if not (n_queries == len(queries) == len(tavily_topics) == len(tavily_days)):
        raise ValueError("The lengths of queries, tavily_topics, and tavily_days must be equal.")
    
    # Web search using async function with corresponding parameters
    search_docs = []
    for query, topic, days in zip(queries, tavily_topics, tavily_days):
        search_doc = tavily_search_sync([query], topic, days)  # Replace with synchronous function
        search_docs.extend(search_doc)  # Add results to the overall list

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000, include_raw_content=True)

    web_results = Document(page_content=source_str,
                           
                        metadata={"source": "Tavily Web Search"} )
    
    #we'll also add to the vectorstore the websearch results so next time it won't need to run web search again for specific questions
    vectorstore.add_documents([web_results])
    
    return {"documents": web_results.page_content}


@traceable
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generated response is grounded in the document and answers the question.
    
    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    # print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    context = state["documents"] #it can be either the retrieved docs or the web search results
    generation = state["generation"]
    
    
    n_iterations = state.get("iterations",None)
    
    if n_iterations!=None:
        n_iterations+=1
    else:
        n_iterations=1
        
        # force_stop = state.get("force_stop",None)
    if n_iterations==2:
        return {"decission":"force_stop", "feedback":None,
                "generation":state["generation"]}
        
        
    generation_grader_prompt = generation_grader_instructions.format(documents=context, generation=generation)
    
    already_searched_queries = state["search_queries_params"]                                                                
   
    answer_grader_prompt = answer_grader_instructions.format(generation=generation, question=question,
                                                             searched_queries=already_searched_queries)
    
    
    messages = [
        {"role": "system", "content": generation_grader_prompt},
        # {"role": "user", "content": f": {}"}
    ]   
    response = json.loads(llm_json.invoke(messages).content)
    
    
        
    # print(grade)
    # print("Generation grader output json:", response)
  
  #----------------  
    if response["score"].lower() == "yes": #the answer is grounded in the documents
        
    
        #now, we check such generation correctly answers the question
        messages = [
            {"role": "system", "content": answer_grader_prompt},
            # {"role": "user", "content": f": {}"}
        ]    
        
        response = json.loads(llm_json.invoke(messages).content)
        # print("answer grader output json:", response)
        # print(type(response))
        
        
        if response["score"].lower() == "yes":
            
            
            # print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS, AND IT ANSWERS THE QUESTION---")
            # return "useful"
            return {"decission":"useful", "feedback":None, "iterations":n_iterations,
                    "generation":generation}
                                #para que una respuesta sea buena
                                #debera acabar siendo = None
                                #por lo que un nodo no tomara feedback que no le corresponde
        
        
        else:
            # print("---DECISION: GENERATION DOES NOT ANSWER THE QUESTION, RUNNING WEBSEARCH---")
            # return "not_useful" #then we simply retry the websearch (only)
            return {"decission":"not_useful", "feedback":response["feedback"],
                    "iterations":n_iterations, "generation":generation}
            
    else:
        #IF THE MODEL HALLUCINATES, WE RE-RUN ONLY THE GENERATION
        
        # print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        # return "not_grounded"
        return {"decission":"not_grounded", "feedback":response["feedback"],
                "iterations":n_iterations, "generation":generation}
    

#-------------------- graph
    
workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("handle_trivial_question", handle_trivial_question)
workflow.add_node("answer_trivial_question", answer_trivial_question) 
workflow.add_node("retrieve", retrieve) 
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("websearch", search_web)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("answer_with_docs", answer_with_docs)
workflow.add_node("grade_generation_v_documents_and_question", grade_generation_v_documents_and_question)

#conditional nodes
def detect_trivial_question(state):
    """
    Determines whether the question is trivial or not

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    question_type = state["question_type"]
    # if question_type == "is_trivial":
    #     return "answer_trivial_question"
    # else:
    #     return "retrieve"
    return question_type


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    # print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        return "use_websearch"
    else:
        return "answer_with_docs"
    


workflow.set_entry_point("handle_trivial_question")

workflow.add_conditional_edges("handle_trivial_question",
                               detect_trivial_question,
                               {"is_trivial": "answer_trivial_question", "not_trivial": "retrieve"})

workflow.add_edge("answer_trivial_question", END)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate, #after running grade_documents, we get the key of websearch to know if we run it or not
    {
        "use_websearch": "generate_queries", #before websearch, we generate the queries through the generate_queries node
        "answer_with_docs": "answer_with_docs",
    },
)

workflow.add_edge("generate_queries", "websearch") 
workflow.add_edge("websearch", "answer_with_docs") #after web search, we go directly to generation
# workflow.add_edge("answer_with_docs", "handle_trivial_question")
workflow.add_edge("answer_with_docs", "grade_generation_v_documents_and_question")


# workflow.add_conditional_edges( #conditional edges
#     "answer_with_docs",
#     grade_generation_v_documents_and_question,
#     {
#         "not_grounded": "answer_with_docs", #if the llm generation grader result is = "not supported", then re-try
#         "useful": END, #if it is useful, then end the process
#         "not_useful": "generate_queries", #if not useful, then re-create queries again given the previous one (if applicable) and run websearch
#     },
# )


def check_generation(state):
    
    
    if state["decission"] == "useful":
        
        return "useful"
    
    elif state["decission"] == "not_useful":
        
        return "not_useful"
    
    elif state["decission"] == "not_grounded":
        
        return "not_grounded"
    
    elif state["decission"] == "force_stop":
        
        print("forcing stop of the graph...")
        
        
        return "force_stop"
    
    
    print("current feedback provided ", state["feedback"])
        
    
workflow.add_conditional_edges( #conditional edges
    "grade_generation_v_documents_and_question",
    check_generation,
    {
        "not_grounded": "answer_with_docs", #if the llm generation grader result is = "not supported", then re-try
        "useful": END, #if it is useful, then end the process
        "not_useful": "generate_queries", #if not useful, then re-create queries again given the previous one (if applicable) and run websearch,
        "force_stop":END
    },
)




from langgraph.checkpoint.sqlite import SqliteSaver
from contextlib import ExitStack

stack = ExitStack()
memory = stack.enter_context(SqliteSaver.from_conn_string(":memory:"))

app = workflow.compile(checkpointer=memory)
#"recursion_limit":60

# app.invoke({"question":"which are the current relationships betwen Venezuela and Spain", "news_summary":text}, thread)["generation"]

#---------------------------------------------------------------------------
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langsmith import traceable
# from langchain_community.tools.tavily_search import TavilySearchResults
# from typing_extensions import TypedDict
# from typing import List
# import os
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# import json
# from langchain.schema import Document
# from langgraph.graph import END, StateGraph


# web_search_tool = TavilySearchResults(k=3)

# ### State

# class GraphState(TypedDict):
#     news_summary : str
#     question : str
#     generation : str
#     web_search : str #whether to run web search or not
#     documents : str #the concatenated list of documents text content (or search results)

# # Check if the directory and files are readable
# directory = './rag_docs'
# # print(os.access(directory, os.R_OK))  # Checks if the directory is readable
# os.chmod('./rag_docs', 0o755)


# markdown_folder_path = "./rag_docs"  # Set the path to your folder
# documents = []

# # Iterate over all .md files in the directory
# for file in os.listdir(markdown_folder_path):
#     if file.endswith('.md'):
#         markdown_path = os.path.join(markdown_folder_path, file)                     #=fast
#         loader = UnstructuredMarkdownLoader(markdown_path, mode="single", strategy="precise")
#         documents.extend(loader.load())  # Add loaded documents to the list

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1200, chunk_overlap=100, add_start_index=True #starting char pos.
#     #size of characters for each chunk
#     #2nd param: will let us have a little portion of the prev. chunk
#     # so in case the key info is in that chunk, we can have a way to get those prev chars if needed 
    
#     #
# )
# all_splits = text_splitter.split_documents(documents)

# #the so-called chroma database
# local_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# persist_dir = "./chroma_db"
# os.makedirs(persist_dir, exist_ok=True)

# vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings,
#                                     persist_directory=persist_dir)


# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3},
#                                     )


# llm = ChatOpenAI(model="gpt-4o", temperature=0) 

# llm_json = ChatOpenAI(
#         model="gpt-4o",
#         temperature = 0,
#         model_kwargs={"response_format": {"type": "json_object"}})


# #---------------------
# #prompts
# generation_instructions = """
#  You are an assistant for question-answering tasks inside a phone call conversation.
 
 
 
#     Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
#     Keep the answer concise optimized to reduce latency, in a spoken language style.
    
#     Question: {question} 
#     Context: {context} 
    
#     You must also keep in mind that you're answering questions related with the news of the week, so you must always respond properly connecting the answer with the spoken topics, for which i'll attach below as a brief summary.
# """

# docs_grader_instructions = """

#     You are a grader assessing relevance 
#     of a retrieved document to a user question. If the document contains keywords related to the user question, 
#     grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
#     Provide the binary score as a JSON with a single key 'score' and no premable or explaination.

#     Here is the retrieved document: \n\n {document} \n\n
#     Here is the user question: {question}
# """

# generation_grader_instructions = """
#     You are a grader assessing whether 
#     an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
#     whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
#     single key 'score' and no preamble or explanation.
#     Here are the facts:
#     \n ------- \n
#     {documents} 
#     \n ------- \n
    
#     Here is the generated answer: \n\n {generation} \n\n
# """

# answer_grader_instructions = """

#     You are a grader assessing whether an 
#     answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
#     useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    
#     Here is the generated answer:
#     \n ------- \n
#     {generation} 
#     \n ------- \n
#     Here is the original user question: {question}
# """

# #--------------- nodes

# @traceable
# def retrieve(state):
#     """
#     Retrieve documents from vectorstore

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     # print("---RETRIEVE---")
#     question = state["question"]

#     # Retrieval
#     retrieved_docs = retriever.invoke(question)
    
#     context = ' '.join([doc.page_content for doc in retrieved_docs])
    
#     return {"documents": context, "question": question}
# #

# @traceable
# def grade_documents(state):
#     """
#     Determines whether the retrieved documents are relevant to the question
#     If any document is not relevant, we will set a flag to run web search

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Filtered out irrelevant documents and updated web_search state
#     """

#     # print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#     question = state["question"]
#     context = state["documents"]
    
    
#     system_instructions = docs_grader_instructions.format(question=question, document=context)
#     messages = [
#         {"role": "system", "content": system_instructions},
#         # {"role": "user", "content": f": {}"}
#     ]   
#     grade = json.loads(llm_json.invoke(messages).content)["score"]
#     # print(grade)
    
#     if grade.lower() == "yes":
#         # print("---GRADE: DOCUMENT RELEVANT---")
#         return {"documents": context, "question": question, "web_search": "No"}
#     else:
#         # print("---GRADE: RETRIEVED DOCUMENTS NOT RELEVANT, RUNNING WEB SEARCH INSTEAD---")
#         return {"documents": context, "question": question, "web_search": "Yes"}
    

# @traceable
# def generate(state):
#     """
#     Generate answer either using RAG on retrieved documents or web search results

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, generation, that contains LLM generation
#     """
#     # print("---GENERATE---")
#     question = state["question"]
#     context = state["documents"]
    
#     system_instructions = generation_instructions.format(question=question, context=context)
    
#     messages = [
#         {"role": "system", "content": system_instructions},
#         {"role": "user", "content": f": Here is the news summary shown to the user {state['news_summary']}"}
#     ]    
#     answer = llm.invoke(messages).content
     
#     return {"documents": context, "question": question, "generation": answer}
# #

# @traceable
# def web_search(state):
#     """
#     Web search based based on the question

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Appended web results to documents
#     """

#     # print("---WEB SEARCH---")
#     question = state["question"]
#     context = state["documents"] if "documents" in state else None

#     # Web search
#     docs = web_search_tool.invoke({"query": question})
#     web_results = "\n".join([d["content"] for d in docs])
#     web_results = Document(page_content=web_results)
#     #we override the past documents, we add the search results
    
#     # if context is not None:
#     #     context += f"\n {web_results.page_content}"
#     # else:
#     #     context = web_results.page_content
    
#     context = web_results.page_content
    
#     return {"documents": context, "question": question}


# @traceable
# def grade_generation_v_documents_and_question(state):
#     """
#     Determines whether the generated response is grounded in the document and answers the question.
    
#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Decision for next node to call
#     """

#     # print("---CHECK HALLUCINATIONS---")
#     question = state["question"]
#     context = state["documents"] #it can be either the retrieved docs or the web search results
#     generation = state["generation"]
    
    
#     generation_grader_prompt = generation_grader_instructions.format(documents=context,
#                                                                      generation=generation)
#     answer_grader_prompt = answer_grader_instructions.format(generation=generation, question=question)
    
    
#     messages = [
#         {"role": "system", "content": generation_grader_prompt},
#         # {"role": "user", "content": f": {}"}
#     ]   
#     grade = json.loads(llm_json.invoke(messages).content)["score"]
#     # print(grade)
    
#     if grade.lower() == "yes": #the answer is grounded in the documents
        
    
#         #now, we check such generation correctly answers the question
#         messages = [
#             {"role": "system", "content": answer_grader_prompt},
#             # {"role": "user", "content": f": {}"}
#         ]    
        
#         grade = json.loads(llm_json.invoke(messages).content)["score"]
        
#         if grade.lower() == "yes":
#             # print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS, AND IT ANSWERS THE QUESTION---")
#             return "useful"
        
        
#         else:
#             # print("---DECISION: GENERATION DOES NOT ANSWER THE QUESTION, RUNNING WEBSEARCH---")
#             return "not_useful" #then we simply retry the websearch (only)
        
#     else:
#         #IF THE MODEL HALLUCINATES, WE RE-RUN ONLY THE GENERATION
        
#         # print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
#         return "not_grounded"
    

# #-------------------- graph
    
# workflow = StateGraph(GraphState)
# # Define the nodes
# workflow.add_node("retrieve", retrieve) 
# workflow.add_node("grade_documents", grade_documents)
# workflow.add_node("websearch", web_search) 
# workflow.add_node("generate", generate)

# #conditional nodes
# @traceable
# def decide_to_generate(state):
#     """
#     Determines whether to generate an answer, or add web search

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Binary decision for next node to call
#     """

#     # print("---ASSESS GRADED DOCUMENTS---")
#     web_search = state["web_search"]

#     if web_search == "Yes":
#         return "websearch"
#     else:
#         return "generate"
    
# workflow.set_entry_point("retrieve")

# workflow.add_edge("retrieve", "grade_documents")



# workflow.add_conditional_edges(
#     "grade_documents",
#     decide_to_generate, #after running grade_documents, we get the key of websearch to know if we run it or not
#     {
#         "websearch": "websearch",
#         "generate": "generate",
#     },
# )

# workflow.add_edge("websearch", "generate") #after web search, we go directly to generation



# workflow.add_conditional_edges( #conditional edges
#     "generate",
#     grade_generation_v_documents_and_question,
#     {
#         "not_grounded": "generate", #if the llm generation grader result is = "not supported", then re-try
#         "useful": END, #if it is useful, then end the process
#         "not_useful": "websearch", #if not useful, then run websearch
#     },
# )

# #--------------- compiling the graph

# #todo: check if short term and long term memmory are useful
# #todo: redefine the node of web search to generate multiple queries given the question (add tavily_days and tavily_topic too)
# from langgraph.checkpoint.sqlite import SqliteSaver
# from contextlib import ExitStack

# stack = ExitStack()
# memory = stack.enter_context(SqliteSaver.from_conn_string(":memory:"))

# app = workflow.compile(checkpointer=memory)

# thread = {"configurable": {"thread_id":"1"}} #"recursion_limit":60

# # app.invoke({"question": "balablabla"}, thread)