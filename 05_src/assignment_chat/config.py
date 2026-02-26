from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Config(BaseModel):
    api_gateway_key: str
    openai_base_url: str
    hn_api_base_url: str
    chat_model: str


config = Config(
    api_gateway_key=os.environ["API_GATEWAY_KEY"],
    hn_api_base_url=os.getenv("HN_API_BASE_URL", "https://hn.algolia.com/api/v1"),
    openai_base_url=os.getenv("OPENAI_BASE_URL", "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"),
    chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
)
