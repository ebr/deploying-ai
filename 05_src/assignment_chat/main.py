import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

from prompts import make_system_message

from llm import llm
from services.hackernews_digest import is_digest_request, fetch_hn_digest
from services.rag import retrieve_rag

tools = []

agent = create_agent(
    llm,
    tools=tools,
)

def chat(message: str, history: list) -> str:
    # Simple case - just return a HN digest for the query. Uses service #1 from the assignment.
    user_wants_hn_digest, maybe_topic = is_digest_request(message)

    if user_wants_hn_digest:
        # all llm-calling is done inside the service #1 itself
        # call into it, and return results back to chat without performing any other queries.
        # for the sake of the assignment documentation: on every call to fetch_hn_digest,
        # we are also embedding (using service #2) the articles and storing in chromadb,
        # so that they can be retrieved later by the RAG tool.

        return fetch_hn_digest(maybe_topic)

    # the user didn't ask for the news digest.
    # do we know anything about their query from our RAG tool? Use service #2 to find out

    rag_result = retrieve_rag(message)
    # rag_result=None

    result = agent.invoke(input={"messages": [make_system_message(), AIMessage(f"{rag_result}"), HumanMessage(message)]})
    return result["messages"][-1].content

app = gr.ChatInterface(fn=chat, title="🍀 HackerNews Evil Leprechaun 🍀")


if __name__ == "__main__":
    app.launch()
