"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""

    # API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model configuration
    LLM_MODEL = "llama-3.3-70b-versatile"  # ✅ valid Groq model

    # Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        if not cls.GROQ_API_KEY:
            raise ValueError("🚨 GROQ_API_KEY not found in .env file")

        # ✅ Correct constructor for Groq LLM
        return ChatGroq(
            api_key=cls.GROQ_API_KEY,
            model=cls.LLM_MODEL,
            temperature=0.2,
        )
