import os
import requests

def run(args):
    platform = args.get("platform", "twitter")
    content = args.get("content", "")
    if not content:
        return "Error: No content to post."

    if platform == "twitter":
        bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not bearer_token:
            return "Twitter not configured. Set TWITTER_BEARER_TOKEN environment variable."
        try:
            url = "https://api.twitter.com/2/tweets"
            headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
            payload = {"text": content[:280]}
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 201:
                return "✅ Tweet posted successfully!"
            else:
                return f"Twitter error: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"Twitter post error: {e}"
    else:
        return f"Platform '{platform}' not supported yet. Only 'twitter'."

def get_info():
    return {
        "name": "Social Media",
        "description": "Posts to social media (Twitter). Arguments: 'platform' (twitter), 'content' (text). Use when user wants to tweet or post on social media."
    }