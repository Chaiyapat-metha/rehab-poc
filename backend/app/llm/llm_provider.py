import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

from app import config

def get_llm():
    """
    Creates and returns an LLM instance based on the application config.
    Also sets up in-memory caching for LLM calls.
    """
    # Load environment variables from .env file (for API keys)
    load_dotenv()

    llm_config = config.app_config.get('llm', {})
    provider = llm_config.get('provider')

    if provider == "openrouter":
        # --- Setup In-Memory KV Cache ---
        # LangChain will automatically cache identical calls to the LLM
        set_llm_cache(InMemoryCache())
        print("⚡ In-memory LLM cache enabled.")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file.")

        llm = ChatOpenAI(
            model=llm_config.get('model_name', 'qwen/qwen3-30b-a3b:free'),
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7, # Controls creativity
            max_tokens=500
        )
        return llm
    else:
        # TODO: Add logic for other providers like OpenAI, local models, etc.
        raise NotImplementedError(f"LLM provider '{provider}' is not supported yet.")
