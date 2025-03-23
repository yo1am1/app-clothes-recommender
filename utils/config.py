import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

_env = load_dotenv(
    dotenv_path=Path(__file__).parent.parent / ".env",
)


@dataclass
class Config:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") if _env else None
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") if _env else None
    AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL") if _env else None
    AZURE_OPENAI_SUBSCRIPTION_KEY = (
        os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY") if _env else None
    )
    EMBEDDING_MODEL = "embedding"  # "gpt-4o"
    CHAT_MODEL = "gpt-4o"
