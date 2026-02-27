import chromadb
from chromadb import EmbeddingFunction, Documents
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config

# TODO use config maybe
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.embed_model,
        api_key=SecretStr(config.api_gateway_key),
        base_url=config.openai_base_url,
        default_headers={"x-api-key": config.api_gateway_key},
    )


class _EmbeddingFunction(EmbeddingFunction):
    # this is needed because chroma's embedding_function interface
    # needs a specific signature, with list[str] as the argument
    def __call__(self, input: Documents):
        return _get_embeddings().embed_documents(list(input))


collection = chroma_client.get_or_create_collection(
    name="hn_articles",
    embedding_function=_EmbeddingFunction(),
)


def embed_article(url: str | None) -> None:
    if url is None:
        # nothing to do
        return

    # load the article content right in here
    # not good practice as this is side effect-y, but i was running out of time
    docs = WebBaseLoader(url).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    collection.add(
        ids=[f"{url}_{i}" for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[{"source": url} for _ in chunks],
    )


def _retrieve(query: str, k: int = 5) -> list[str]:
    # retrieve relevant chunks from chroma
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0] if results["documents"] else []


@tool
def retrieve_rag(query: str) -> str:
    """Retrieve relevant HackerNews content from the local vector store using semantic search."""
    chunks = _retrieve(query)
    if not chunks:
        return "No relevant content found."
    return "\n\n---\n\n".join(chunks)
