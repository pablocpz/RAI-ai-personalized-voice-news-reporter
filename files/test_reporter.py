from utils.reporter_graph import graph as reporter_graph #Reporter Graph
from utils.news_reports import run_reports_creation
from utils.news_reports import get_news_data
import asyncio

async def main():
    picked_headlines, news_content = get_news_data()

    reports_list = await run_reports_creation(reporter_graph,
                                            picked_headlines=picked_headlines,
                                            news_content=news_content)
    
    #await needs to be wrapped inside another external async function
    
if __name__ == "__main__":
    # Run the async function with asyncio
    asyncio.run(main())
    