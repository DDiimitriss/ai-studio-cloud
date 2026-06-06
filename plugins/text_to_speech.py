import os
import urllib.parse

def run(args):
    """Convert text to speech"""
    text = args.get("text", "")
    
    if not text:
        return "❌ No text provided. Tell me what to say!"
    
    # Create audio URL
    encoded_text = urllib.parse.quote(text)
    audio_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl=en&client=tw-ob"
    
    # Return a simple markdown link
    return f"🔊 **[Click here to hear the audio]({audio_url})**\n\n(Or copy this URL into a new tab: {audio_url})"

def get_info():
    return {
        "name": "Text to Speech",
        "description": "Converts text to speech and returns an audio link. Arguments: 'text' (the text to speak). Use when user wants to hear the response spoken aloud."
    }