from typing import List
from pydantic import BaseModel
from utils.news_tools import retrieve_weekly_news,retrieve_news_content
from langchain_openai import ChatOpenAI
import asyncio

    
def get_news_data():
    
    gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0)


    selector_instructions = """

    You will receive a list of Spanish headlines from a weekly journal. Each headline is a string in the list. Your task is to filter and select only the headlines that cover the following topics of interest: **politics, national or international affairs, technology, science, and economy**.

    Follow these instructions:
    1. Carefully read each headline and determine its topic.
    2. Only select headlines that match the specified topics of interest. Headlines must be about actual news events, active problems and developments, not opinion pieces or analyses.
    3. Disregard headlines about entertainment, sports, lifestyle, or other unrelated categories.
    4. Identify and remove duplicates. If two or more headlines refer to the same news event or topic, keep only one representative headline.
    5. Output the selected headlines as a Python list of strings.

    - Input Example:
    [
        'Ni cinta de correr, ni bicicleta estática: este es el entrenamiento que más adelgaza y con el que seguirás quemando calorías hasta 24 horas después de haber terminado la sesión (incluso, durmiendo) - El Mundo',
        'Eva Arguiñano, la pastelera que aprendió repostería a marchas forzadas y llorando mucho - El Mundo',
        "Un artículo de Elon Musk en el 'Die Welt' apoyando a la extrema derecha alemana provoca la renuncia de su responsable de Opinión - El Mundo",
        'El jefe de la OMS escapó por poco de morir durante los bombardeos israelíes en el aeropuerto de Yemen: "El ruido era ensordecedor. Todavía me zumban los oídos" - El Mundo',
        "La 'flota fantasma' de barcos mercantes que permite a Putin realizar sabotajes, exportar petróleo y expandir el poder ruso - El Mundo"
    ]

    - Output Example:**  
    [
        'El jefe de la OMS escapó por poco de morir durante los bombardeos israelíes en el aeropuerto de Yemen: "El ruido era ensordecedor. Todavía me zumban los oídos" - El Mundo',
        "Un artículo de Elon Musk en el 'Die Welt' apoyando a la extrema derecha alemana provoca la renuncia de su responsable de Opinión - El Mundo",
    ]

    """

    class HeadlineList(BaseModel):
        headlines: List[str]
        
    
    weekly_news_json = retrieve_weekly_news()

    news_headlines = [article["title"] for article in weekly_news_json]



    structured_llm = gpt_4o.with_structured_output(HeadlineList)

    picked_headlines = structured_llm.invoke([
            
        {"role":"system", 'content':selector_instructions},
            
        {"role":"user", "content":f"these are the news headlines: {news_headlines}"}])



    news_content = retrieve_news_content(news_headlines=picked_headlines.headlines)

    return picked_headlines, news_content


import asyncio

async def process_graph(headline, page_content, graph):
    """
    Asynchronously runs the graph for a single pair of headline and page content.
    """
    report_state = {
        "headline": headline,
        "page_content": page_content,
        "sections": [],  # Empty list for sections
        "completed_sections": [],  # Empty list for completed sections
        "report_sections_from_research": "",  # Empty string for research sections
        "final_report": ""  # Empty string for final report
    }
    return await graph.ainvoke(report_state)

async def main(picked_headlines, news_content, graph):
    """
    Processes all headline-content pairs in parallel and stores the results in a list.
    """
    tasks = [
        process_graph(headline, page_content, graph)
        for headline, page_content in zip(picked_headlines.headlines, news_content)
    ]
    results = await asyncio.gather(*tasks)
    return results

# Handling the event loop when it's already running
async def run_reports_creation(graph, picked_headlines, news_content):
    """
    takes the reporter graph, runs a pipeline of weekly news selection
    
    news of interest for the user
    """
    return await main(picked_headlines, news_content, graph)



#TODO +++++++++++++++++++SCRIPT

## Call this from an existing async context
# results = await run_in_existing_loop(picked_headlines, news_content, graph)


# from news_tools import export_markdown_reports

# export_markdown_reports(data_list=results, headlines=picked_headlines.headlines, output_folder="rag_docs")