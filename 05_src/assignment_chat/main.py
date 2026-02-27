import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from prompts import prompt_base_system_message

from llm import llm
from services.hackernews_digest import is_digest_request, fetch_hn_digest

# tools = []

agent = create_agent(
    llm,
)


def chat(message: str, history: list) -> str:
    # Simple case - just return a HN digest for the query. Uses service #1 from the assignment.
    is_digest, maybe_topic = is_digest_request(message)
    if is_digest:
        # all llm-calling is done inside the service itself
        # branch out into it, and return without performing any other queries.
        return fetch_hn_digest(maybe_topic)

    result = agent.invoke({"messages": [SystemMessage(prompt_base_system_message), HumanMessage(message)]})
    return result["messages"][-1].content


app = gr.ChatInterface(fn=chat, title="HackerNews RAG Chat")


if __name__ == "__main__":
    app.launch()
