from langchain.prompts import PromptTemplate
# from langchain.runnables.base import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

# Step 1: Instantiate the GPT-4 model from OpenAI using langchain
gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0)

# Step 2: Define a prompt template to summarize each news item in a simple and engaging way
summary_prompt = PromptTemplate(
    input_variables=["news_item"],
    template="Summarize this news story in a short, simple, and engaging way, focusing on the key events. Exclude sports, health, or fashion topics. Use clear and simple language, and explain any political or economic terms in a way that’s easy to understand for someone without a deep background in these areas.\\n{news_item}"
)

# Step 3: Define a template for the final narrative
final_narrative_prompt = PromptTemplate(
    input_variables=["summaries"],
    template=(
        "You are a friendly and engaging presenter. Your goal is to explain the important news events that happened this week in a brief, simple, and fun way. "
        "The following are the summaries of each important event:\\n{summaries}\\n"
        "Now, weave these events together and tell the story in a short, high-level overview. Start with an engaging phrase like 'Ready to know what happened this week?'"
        "Focus on making the information digestible and explain any complex concepts in a simple, clear manner."
    )
)

# Step 4: Define the chain for summarizing the news items
summary_chain = summary_prompt | gpt_4o | StrOutputParser()

# Step 5: Define the chain for generating the final narrative (DEFINE THE PIPELINE)
final_narrative_chain = final_narrative_prompt | gpt_4o | StrOutputParser()

#todo: here we give to the llm the final narrative prompt, and we parse the output so it is in STRING format

# Step 6: Function to summarize each news item in a simple and short way
def summarize_news(news_list):
    summaries = []
    for news_item in news_list:
        summary = summary_chain.invoke({"news_item": news_item})  # Using the new invoke method for processing
        summaries.append(summary)
    return summaries

# Step 7: Function to create a final engaging narrative combining all summaries
def create_final_narrative(summaries):
    summaries_str = "\\n".join([f"Event {i+1}: {summary}" for i, summary in enumerate(summaries)])
    final_narrative = final_narrative_chain.invoke({"summaries": summaries_str})  # Generate the final narrative
    return final_narrative

# Step 8: Function to read and process the news reports from files
# def load_news_reports(directory_path="rag_docs"):
#     #directory from where we call this script
#     news_list = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".md"):
#             with open(os.path.join(directory_path, filename), 'r') as file:
#                 news_list.append(file.read())
#     return news_list

import os

def load_news_reports(directory_path="rag_docs"):
    # directory from where we call this script
    news_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            try:
                # Try opening the file with UTF-8 encoding
                with open(file_path, 'r', encoding='utf-8') as file:
                    news_list.append(file.read())
            except UnicodeDecodeError:
                # Handle the case where decoding fails
                print(f"Warning: Could not decode file {file_path} with UTF-8 encoding.")
                with open(file_path, 'r', encoding='latin1', errors='replace') as file:
                    news_list.append(file.read())
    return news_list


# Step 9: Main function that orchestrates summarizing the news and creating the final narrative
def get_news_narrative():
    # Step 1: Load news reports
    news_list = load_news_reports()
    
    # Step 2: Summarize the news stories
    summaries = summarize_news(news_list)
    
    # Step 3: Generate the final engaging narrative
    final_narrative = create_final_narrative(summaries)
    
    # Step 4: Return or deliver the final story
    return final_narrative