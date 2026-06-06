import os
import requests
import base64

def run(args):
    """Convert text to speech using OpenRouter's TTS"""
    text = args.get("text", "")
    voice = args.get("voice", "alloy")  # alloy, echo, fable, onyx, nova, shimmer
    
    if not text:
        return "❌ No text provided. Tell me what to say!"
    
    # For now, we'll use a simple browser-based approach
    # Return JavaScript that will use the browser's built-in speech synthesis
    js_code = f"""
    <button onclick="speakText()" style="background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
        🔊 Play Audio
    </button>
    <script>
    function speakText() {{
        const text = {repr(text)};
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        speechSynthesis.speak(utterance);
    }}
    </script>
    """
    
    return js_code

def get_info():
    return {
        "name": "Text to Speech",
        "description": "Converts text to speech and provides a play button. Arguments: 'text' (the text to speak), 'voice' (optional: alloy, echo, fable, onyx, nova, shimmer). Use when user wants to hear the response spoken aloud."
    }