import sys
import asyncio
from typing import Any, cast
import httpx
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.insert(
    0, str(Path(__file__).parents[1])
)  # need this because this isn't a proper module
from config import config
from llm import llm

### For service 1 - HN digest - we aren't using a tool, but simply returning the output of the sevrice
class DigestIntent(BaseModel):
    # Use our LLM to determine whether the user is asking for a hackernews digest or not.
    # yes, we could potentially use a keyword-based approach here, but we like to throw AI at the problem.
    is_digest_request: bool = Field(description="Is the user asking for a HackerNews digest?")
    topic: str | None = Field(default=None, description="What topic, if any, is the user asking about? Just return the topic, no extra words. If not a digest request, do not return a value.")

def is_digest_request(query: str) -> tuple[bool, str | None]:
    # truncating for safety bc we don't want to blow the context if the user pastes in "war and peace"
    truncated = query[:100]

    _digest_classifier = llm.with_structured_output(DigestIntent)

    _digest_system = SystemMessage(
        "Is the user asking for a HackerNews digest about a specific topic?"
        "Respond with structured output."
    )

    result = cast(DigestIntent, _digest_classifier.invoke([_digest_system, HumanMessage(truncated)])) # cast just to placate the type checker
    return result.is_digest_request, result.topic


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


def fetch_hn_digest(query: str | None) -> Any:  # will deal with typing someday i'm sure
    """
    Fetch recent HackerNews articles matching the query and summarize the top stories.
    """
    articles = asyncio.run(
        _get_hn_articles(HackerNewsQueryParams(query=query or "", tags=["story"]))
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
        f"Title: {a.title}\nAuthor: {a.author}\nDate: {a.date}\nText: {a.text or 'N/A'} \n\nLink: https://news.ycombinator.com/item?id={a.story_id}"
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
    results = fetch_hn_digest(query)
    print(results)
