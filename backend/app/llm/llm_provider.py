# File: C:\Users\chaiyapat metha\Desktop\AI Project\rehab-poc\backend\app\llm\llm_provider.py 

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

from app.config import load_config
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    """
    Creates and returns an LLM instance based on the application config.
    Also sets up in-memory caching for LLM calls.
    """
    # Load environment variables from .env file (for API keys)
    load_dotenv()
    
    full_config = load_config() 
    llm_config = full_config.get('llm', {}) 

    provider = llm_config.get('provider')

    if provider == "openrouter":
        # --- Setup In-Memory KV Cache (Non-real-time RAG) ---
        set_llm_cache(InMemoryCache())
        print("⚡ In-memory LLM cache enabled.")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file. Please check your .env setup.")

        llm = ChatOpenAI(
            model=llm_config.get('model_name', 'qwen/qwen3-30b-a3b:free'),
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7, 
            max_tokens=500
        )
        return llm
    else:
        if not provider:
            raise NotImplementedError("LLM provider is not configured in models.yaml under 'llm' section.")
        else:
            raise NotImplementedError(f"LLM provider '{provider}' is not supported yet.")
