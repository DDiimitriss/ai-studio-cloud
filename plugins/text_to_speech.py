import os
import requests
import base64
import urllib.parse

def run(args):
    """Convert text to speech using a free TTS service"""
    text = args.get("text", "")
    
    if not text:
        return "❌ No text provided. Tell me what to say!"
    
    # Use a free TTS service that returns audio
    # We'll use Google Translate's TTS (free, no API key needed)
    encoded_text = urllib.parse.quote(text)
    audio_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl=en&client=tw-ob"
    
    # Return HTML5 audio player
    html = f"""
<div style="margin: 10px 0; padding: 15px; background: #f0f0f0; border-radius: 8px;">
    <p style="margin: 0 0 10px 0; font-weight: bold;">🔊 Audio Response:</p>
    <audio controls style="width: 100%;">
        <source src="{audio_url}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">Click play to hear the response</p>
</div>
"""
    
    return html

def get_info():
    return {
        "name": "Text to Speech",
        "description": "Converts text to speech and provides an audio player. Arguments: 'text' (the text to speak). Use when user wants to hear the response spoken aloud."
    }