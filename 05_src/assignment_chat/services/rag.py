import hashlib

import chromadb
from chromadb import EmbeddingFunction, Documents
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
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

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

def embed_article(docs: list | None) -> None:
    if docs is None:
        # nothing to do
        return

    chunks = splitter.split_documents(docs)
    collection.upsert( # upsert so we don't get duplicate id errors
        # chunk ids based on hash of page content, to avoid duplicates
        ids=[hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks],
        documents=[chunk.page_content for chunk in chunks],
    )


def _retrieve_from_vectorstore(query: str, k: int = 5) -> list[str]:
    # retrieve relevant chunks from chroma
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0] if results["documents"] else []


def retrieve_rag(query: str) -> str | None:
    """Retrieve relevant content from the vector store using semantic search."""
    chunks = _retrieve_from_vectorstore(query)
    if not chunks:
        return None
    return "\n\n---\n\n".join(chunks)
