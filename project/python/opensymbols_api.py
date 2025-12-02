import requests
from config import BASE_URL, TOKEN_URL, SHARED_SECRET

_access_token = None

def get_access_token():
    """Generate access token from shared secret."""
    global _access_token
    if _access_token:
        return _access_token
    
    r = requests.post(TOKEN_URL, json={"secret": SHARED_SECRET}, timeout=10)
    r.raise_for_status()
    _access_token = r.json().get("access_token")
    return _access_token

def query_symbol(word, preferred_source=None):
    """Query OpenSymbols API for a word."""
    token = get_access_token()
    params = {
        "q": word,
        "locale": "en",
        "access_token": token
    }
    r = requests.get(
        BASE_URL,
        params=params,
        timeout=10
    )
    r.raise_for_status()
    results = r.json()

    if not results:
        return None

    if preferred_source:
        filtered = [s for s in results if s.get("repo_key", "").lower() == preferred_source.lower()]
        if filtered:
            return filtered[0]

    return results[0]  # fallback to first result
