import os
import requests
from duckduckgo_search import DDGS

def run(args):
    query = args.get("query", "")
    engine = args.get("engine", "ddg")

    if not query:
        return "Error: No search query provided."

    # DuckDuckGo (no key needed)
    if engine == "ddg":
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if not results:
                    return "No results found."
                formatted = "\n\n".join([
                    f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                    for r in results
                ])
                return formatted
        except Exception as e:
            return f"DuckDuckGo search failed: {e}"

    # Google (requires API key and CSE ID)
    elif engine == "google":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        cse_id = os.environ.get("GOOGLE_CSE_ID", "")
        if not api_key or not cse_id:
            return "Google search not configured. Set GOOGLE_API_KEY and GOOGLE_CSE_ID."
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": api_key, "cx": cse_id, "q": query, "num": 5}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if "items" not in data:
                return "No Google results."
            formatted = "\n\n".join([
                f"Title: {item['title']}\nURL: {item['link']}\nSnippet: {item.get('snippet', '')}"
                for item in data["items"]
            ])
            return formatted
        except Exception as e:
            return f"Google search error: {e}"

    # Bing (requires API key)
    elif engine == "bing":
        api_key = os.environ.get("BING_API_KEY", "")
        if not api_key:
            return "Bing search not configured. Set BING_API_KEY."
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {"q": query, "count": 5}
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            data = resp.json()
            if "webPages" not in data:
                return "No Bing results."
            formatted = "\n\n".join([
                f"Title: {item['name']}\nURL: {item['url']}\nSnippet: {item.get('snippet', '')}"
                for item in data["webPages"]["value"]
            ])
            return formatted
        except Exception as e:
            return f"Bing search error: {e}"

    else:
        return f"Unknown engine '{engine}'. Use 'ddg', 'google', or 'bing'."

def get_info():
    return {
        "name": "Multi Search",
        "description": "Searches the web using DuckDuckGo, Google, or Bing. Arguments: 'query', 'engine' (ddg, google, bing – default ddg)."
    }