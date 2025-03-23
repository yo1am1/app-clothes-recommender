from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from utils.config import Config
import logging


def __get_openai_embeddings() -> AzureOpenAIEmbeddings:
    """Create an instance of AzureOpenAIEmbeddings.

    Returns:
        AzureOpenAIEmbeddings: An instance of AzureOpenAIEmbeddings.
    """
    if not all(
        [
            Config.AZURE_OPENAI_API_KEY,
            Config.AZURE_OPENAI_API_VERSION,
            Config.AZURE_OPENAI_BASE_URL,
            Config.AZURE_OPENAI_SUBSCRIPTION_KEY,
        ]
    ):
        logging.warning("Missing Azure OpenAI configuration. Chatbot is not available.")
        return None

    try:
        _emb = AzureOpenAIEmbeddings(
            openai_api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            base_url=Config.AZURE_OPENAI_BASE_URL,
            default_headers={
                "Ocp-Apim-Subscription-Key": Config.AZURE_OPENAI_SUBSCRIPTION_KEY
            },
            model=Config.EMBEDDING_MODEL,
        )
    except Exception as e:
        logging.error(f"Error creating OpenAI embeddings: {e}")
        _emb = None

    return _emb


def __get_openai_model(
    model: str = Config.CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 30,
    max_retries: int = 2,
) -> AzureChatOpenAI:
    """Create an instance of AzureChatOpenAI.

    Args:
        model (str, optional): The model to use. Defaults to Config.CHAT_MODEL.
        temperature (float, optional): The temperature. Defaults to 0.2.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 4096.
        timeout (int, optional): The timeout. Defaults to 30.
        max_retries (int, optional): The maximum number of retries. Defaults to 2.

    Returns:
        AzureChatOpenAI: An instance of AzureChatOpenAI.
    """
    # Check for required credentials
    if not all(
        [
            Config.AZURE_OPENAI_API_KEY,
            Config.AZURE_OPENAI_API_VERSION,
            Config.AZURE_OPENAI_BASE_URL,
            Config.AZURE_OPENAI_SUBSCRIPTION_KEY,
        ]
    ):
        logging.warning("Missing Azure OpenAI configuration. Chatbot is not available.")
        return None

    try:
        _model = AzureChatOpenAI(
            model_name=model,
            openai_api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            base_url=Config.AZURE_OPENAI_BASE_URL,
            default_headers={
                "Ocp-Apim-Subscription-Key": Config.AZURE_OPENAI_SUBSCRIPTION_KEY
            },
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as e:
        logging.error(f"Error creating OpenAI model: {e}")
        _model = None

    return _model
