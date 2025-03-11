import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).parent.parent / ".env",
)


@dataclass
class Config:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL")
    AZURE_OPENAI_SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
    EMBEDDING_MODEL = "embedding"
    CHAT_MODEL = "gpt-4o"
