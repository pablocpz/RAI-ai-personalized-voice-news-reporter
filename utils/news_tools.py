import os
import re
from datetime import datetime, timedelta
from gnews import GNews  # Assuming GNews is already imported
import requests
import json
import os
from tavily import TavilyClient

from typing import List
from langchain_openai import ChatOpenAI
from typing import Union

gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0) 

tavily_client = TavilyClient()

def check_and_select_url(organic_results, news_headlines):
    try:
        response = gpt_4o.invoke([{
            "role": "system", "content": """
                You are a helpful assistant. Your task is to select the most relevant URL from the provided organic search results based on the given news headline.
                
                Specific instructions:
                - You should only consider URLs from trustworthy news sources such as 'elmundo.es', 'elpais.com', 'bbc.com', 'nytimes.com', 'reuters.com', etc.
                - Exclude social media platforms like Twitter, Instagram, Facebook, or YouTube.
                - The 'organic_results' is a list of dictionaries, each containing a 'title' and 'link'. Only consider these keys.
                - **Before selecting a URL, make sure that the content in the URL is highly relevant to the provided headline.** Consider whether the topic, key phrases, and overall focus of the article align with the headline.
                - Only return the URL that best matches the headline. If you find no relevant matches, return an empty string.
                - Do not create or infer URLs. Only return the URLs that are provided in the 'organic_results' with no preamble or explanations, just the URL itself.
            """
        },
        {
            "role": "user", "content": f"These are the news headline: \n\n {news_headlines}"
        },
        {
            "role": "user", "content": f"Here are the organic results: \n\n {organic_results}"
        }
        ])

        # Extract the URL from the LLM response
        best_url = response.content
        return best_url

    except Exception as e:
        return f"[ * ] Error in check_and_select_url: {str(e)}"


def format_results(organic_results, news_headlines):
    if not organic_results:
        print("[ * ] No organic results found.")  # Debug: No organic results at all
        return "[ * ] No results from elmundo.es found."  # If no results are found

    # Debug: Show all links being checked
    print(f"Checking organic results: {len(organic_results)} results found.")
    
    # Check if any link from elmundo.es is found
    for idx, result in enumerate(organic_results):
        link = result.get('link', '#')  # Extract the link of the article
        print(f"Checking result {idx + 1}: {link}")  # Debug: Display the current link being checked

        # Check if the link is from elmundo.es and return it if it's a match
        if "elmundo.es" in link:
            print(f"Found elmundo link: {link}")  # Debug: If elmundo link is found
            return link  # Return the first valid link found

    # If no elmundo.es link is found, call the check_and_select_url function
    print("No elmundo.es links found. Calling LLM to find the best match...")
    best_url = check_and_select_url(organic_results, news_headlines)

    # Return the best matching URL or a fallback message if no match is found
    return best_url if best_url != "" else "[ * ] No matching result found."


def get_google_serper(search_term):
    search_url = "https://google.serper.dev/search"  # Endpoint for search
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': os.environ['SERPER_API_KEY']  # Ensure this environment variable is set with your API key
    }
    payload = json.dumps({"q": search_term})
    
    # Attempt to make the HTTP POST request
    try:
        print(f"Sending request for: {search_term}")  # Debug: Search term being used
        response = requests.post(search_url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4XX, 5XX)
        
        results = response.json()
        print(f"Received response: {results}")  # Debug: Full response
        
        # Get the formatted URL using the function
        formatted_results = check_and_select_url(organic_results=results.get('organic', []),
                                                  news_headlines=search_term)  # Get the URL based on the headlines
        
        return formatted_results  # Return the selected URL or message

    except requests.exceptions.HTTPError as http_err:
        print(f"[ * ] HTTP error occurred: {http_err}")  # Debug: HTTP error details
        return f"[ * ] HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as req_err:
        print(f"[ * ] Request error occurred: {req_err}")  # Debug: Request error details
        return f"[ * ] Request error occurred: {req_err}"
    except KeyError as key_err:
        print(f"[ * ] Key error occurred: {key_err}")  # Debug: Key error details
        return f"[ * ] Key error occurred: {key_err}"


def retrieve_weekly_news() -> List:
    """
    retrieves a list of max_results=20 JSON objects with fields: 'title', 'description', 'published date'
    """
    # Calculate the start and end dates for the last 7 days
    end_date = datetime.now()  # Current date
    start_date = end_date - timedelta(days=7)  # 7 days ago

    # Initialize GNews with date range parameters as datetime objects
    google_news = GNews(language="es", start_date=start_date, end_date=end_date,
                        country="ES", max_results=30, period="7d")
    
    #start by retrieving news from ElMundo (spanish journal which also covers international topics)
    
    elmundo_news = google_news.get_news_by_site(site="elmundo.es") #json format
    
    if len(elmundo_news) > 0:
        """
        silly format checking for duplicates
        """
        news = []
        seen = set()

        for obj in elmundo_news:
            obj_tuple = json.dumps(obj, sort_keys=True)
            if obj_tuple not in seen:
                seen.add(obj_tuple)
                news.append(obj)

        return news
    else:
        return "[ * ] Error fetching news"


def retrieve_news_content(news_headlines: List[str]) -> Union[List[str], str]:
    """
    Fetches the content of news articles based on their headlines.

    Args:
        news_headlines (List[str]): A list of news headlines to search for content.

    Returns:
        Union[List[str], str]: A list of article contents if successful, or an error message if fetching fails.
    """
    # Generate URLs for each headline
    try:
        articles_urls = [get_google_serper(search_term=headline) for headline in news_headlines]
        print(articles_urls)
    except Exception as e:
        return f"[ * ] Error generating URLs: {str(e)}"

    try:
        # Slice the first 15 URLs and handle them
        first_batch = articles_urls[:15]
        remaining_batch = articles_urls[15:]

        # Extract content for the first batch (15 URLs)
        print("Extracting first 15 URLs")
        first_batch_content = tavily_client.extract(urls=first_batch)
        print(f"First batch content: {first_batch_content}")

        # Extract content for the remaining batch
        if remaining_batch:
            print("Extracting remaining URLs")
            remaining_batch_content = tavily_client.extract(urls=remaining_batch)
            print(f"Remaining batch content: {remaining_batch_content}")
        else:
            remaining_batch_content = []

        # Combine results from both batches
        extracted_content = []

        if "results" in first_batch_content:
            extracted_content.extend([response.get("raw_content", "") for response in first_batch_content["results"]])
        if "results" in remaining_batch_content:
            extracted_content.extend([response.get("raw_content", "") for response in remaining_batch_content["results"]])

        # Validate that we have the expected number of results
        if len(extracted_content) == len(news_headlines):
            return extracted_content
        else:
            return f"[ * ] Error: Content fetching failed or mismatched result count."

    except Exception as e:
        return f"[ * ] Error fetching content: {str(e)}"
    
    
def format_title(title):
    # Replace special characters with underscores
    title = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces and hyphens with a single underscore
    title = re.sub(r'[\s-]+', '_', title)
    # Limit the length of the title for file naming (optional, adjust as needed)
    max_length = 255  # Adjust this to your needs
    title = title[:max_length]
    return title

def export_markdown_reports(data_list, headlines, output_folder="reports"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ensure both lists have the same length
    if len(data_list) != len(headlines):
        raise ValueError("The lengths of data_list and headlines must match.")

    for index, (item, headline) in enumerate(zip(data_list, headlines)):
        try:
            # Check if the item exists and has a 'final_report' key
            if item is None or 'final_report' not in item:
                print(f"Skipping index {index}: No valid 'final_report' found.")
                continue

            # Extract the report content
            report_content = item['final_report']
            
            # Ensure the report content is not None
            if report_content is None:
                print(f"Skipping index {index}: 'final_report' is None.")
                continue

            # Create a safe filename from the headline
            safe_filename = f"{format_title(headline)}.md"
            
            # Full file path
            file_path = os.path.join(output_folder, safe_filename)

            # Write the report content to a file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(report_content)
            print(f"Exported: {file_path}")

        except Exception as e:
            print(f"Error at index {index}: {e}")