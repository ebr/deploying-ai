"""
Singleton LLM instance(s) for the application
"""

from langchain.chat_models import init_chat_model

from config import config

llm = init_chat_model(
    model=config.chat_model,
    api_key=config.api_gateway_key,
    base_url=config.openai_base_url,
    default_headers={"x-api-key": config.api_gateway_key},
)
