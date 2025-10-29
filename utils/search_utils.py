import os
import sys
from duckduckgo_search import DDGS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import MAX_SEARCH_RESULTS


def search_web(query, max_results=None):
    """
    Perform web search using DuckDuckGo
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string with search results
    """
    max_results = max_results or MAX_SEARCH_RESULTS
    
    try:
        print(f"Searching web for: {query}")
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return "No search results found."
        
        # Format results
        formatted_results = " **Web Search Results:**\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_results += f"**{i}. {result['title']}**\n"
            formatted_results += f"{result['body'][:300]}...\n"
            formatted_results += f"Source: {result['href']}\n\n"
        
        print(f"Found {len(results)} search results")
        return formatted_results
    
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        print(f"✗ {error_msg}")
        return f"Unable to perform web search: {error_msg}"