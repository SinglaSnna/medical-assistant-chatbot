import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# API Keys - Try Streamlit secrets first, then environment variables
try:
    # For Streamlit Cloud deployment
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    # For local development
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model names - UPDATED WITH CURRENT MODELS
GROQ_MODEL = "llama-3.3-70b-versatile"
GOOGLE_MODEL = "gemini-1.5-flash"
OPENAI_MODEL = "gpt-3.5-turbo"

# Embedding model for RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG settings
MAX_RAG_CHUNKS = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Search settings
MAX_SEARCH_RESULTS = 3