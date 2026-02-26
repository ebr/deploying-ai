import sys
import asyncio
from typing import Any
import httpx
from pathlib import Path
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

sys.path.insert(
    0, str(Path(__file__).parents[1])
)  # need this because this isn't a proper module
from config import config


class HackerNewsQueryParams(BaseModel):
    query: str
    tags: list[str] = []
    page: int = 1  # assuming this is 1-indexed, not actually sure tbh
    hits_per_page: int = 10


class HackerNewsArticle(BaseModel):
    story_id: int
    author: str
    title: str
    text: str | None
    date: str | None = None


async def _get_hn_articles(
    query_params: HackerNewsQueryParams,
) -> list[HackerNewsArticle]:
    """
    Gets the top most recent HackerNews stories matching the query parameters.
    """
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        url = f"{config.hn_api_base_url}/search_by_date"
        params = {
            "query": query_params.query,
            "tags": query_params.tags,
            "page": query_params.page,
            "hitsPerPage": query_params.hits_per_page,
        }
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        articles = [
            {
                "story_id": item.get("story_id"),
                "author": item.get("author"),
                "title": item.get("title"),
                "text": item.get("story_text"),
                "date": item.get("created_at"),
            }
            for item in resp.json().get("hits", [])
        ]

        return [HackerNewsArticle(**story_data) for story_data in articles]


def hackernews_digest(query: str) -> Any:  # will deal with typing someday i'm sure
    """
    Fetch recent HackerNews articles matching the query and summarize the top stories.
    """
    articles = asyncio.run(
        _get_hn_articles(HackerNewsQueryParams(query=query, tags=["story"]))
    )

    llm = init_chat_model(
        model=config.chat_model,
        api_key=config.api_gateway_key,  # can be empty bc we use the header, but we'll experiment with other providers so might as well use it
        base_url=config.openai_base_url,
        default_headers={"x-api-key": config.api_gateway_key},
    )

    system_prompt = (
        "You are a helpful assistant that summarizes HackerNews stories."
        "Identify the 4-5 most important stories and explain why each is significant."
        "Provide a one-sentence summary and a two-paragraph analysis for each key story, highlighting the main points and implications."
        "Mention the author and the date of each story"
        "Ensure that the analysis includes some opinion on the potential impact of the story on the relevant industry, geopolitics, and broader society."
        "Finish each entry with a link to the original story."
    )

    articles_text = "\n\n".join(
        f"Title: {a.title}\nAuthor: {a.author}\nDate: {a.date}\nText: {a.text or 'N/A'} \nLink: https://news.ycombinator.com/item?id={a.story_id}"
        for a in articles
    )
    user_message = (
        f"Here are recent HackerNews articles about '{query}':\n\n{articles_text}\n\n"
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )
    return response.content


# testing
if __name__ == "__main__":
    query = "artificial intelligence"
    results = hackernews_digest(query)
    print(results)
