import os
import sys
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY
from config.config import GROQ_MODEL, GOOGLE_MODEL, OPENAI_MODEL


def get_chatgroq_model(api_key=None):
    """Initialize and return the Groq chat model"""
    try:
        key = api_key or GROQ_API_KEY
        if not key:
            raise ValueError("Groq API key not found")
        
        groq_model = ChatGroq(
            api_key=key,
            model=GROQ_MODEL,
            temperature=0.7
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")


def get_openai_model(api_key=None):
    """Initialize and return the OpenAI chat model"""
    try:
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OpenAI API key not found")
        
        openai_model = ChatOpenAI(
            api_key=key,
            model=OPENAI_MODEL,
            temperature=0.7
        )
        return openai_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")


def get_google_model(api_key=None):
    """Initialize and return the Google Gemini chat model"""
    try:
        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError("Google API key not found")
        
        google_model = ChatGoogleGenerativeAI(
            google_api_key=key,
            model=GOOGLE_MODEL,
            temperature=0.7
        )
        return google_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Google model: {str(e)}")


def get_model(provider="groq", api_key=None):
    """Get model based on provider selection"""
    provider = provider.lower()
    
    if provider == "groq":
        return get_chatgroq_model(api_key)
    elif provider == "openai":
        return get_openai_model(api_key)
    elif provider == "google":
        return get_google_model(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")