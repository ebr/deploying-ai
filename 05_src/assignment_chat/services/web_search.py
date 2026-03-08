from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=10,
    topic="general"
)