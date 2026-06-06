def run(args):
    """Convert text to speech using browser's built-in speech synthesis"""
    text = args.get("text", "")
    voice_type = args.get("voice", "default")  # default, fast, slow
    
    if not text:
        return "❌ No text provided. Tell me what to say!"
    
    # Determine speech rate
    if voice_type == "fast":
        rate = 1.5
    elif voice_type == "slow":
        rate = 0.8
    else:
        rate = 1.0
    
    # Return JavaScript that will speak when the page loads
    # This uses the browser's built-in Web Speech API
    js_speech = f"""
<script>
(function() {{
    if ('speechSynthesis' in window) {{
        const utterance = new SpeechSynthesisUtterance({repr(text)});
        utterance.rate = {rate};
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // Try to find a good English voice
        const voices = speechSynthesis.getVoices();
        const englishVoice = voices.find(v => v.lang.startsWith('en'));
        if (englishVoice) {{
            utterance.voice = englishVoice;
        }}
        
        speechSynthesis.speak(utterance);
        console.log('🔊 Speaking...');
    }} else {{
        console.log('Browser does not support speech synthesis');
    }}
}})();
</script>
✅ **Audio is playing!** (If you don't hear anything, check your speaker volume)
"""
    
    return js_speech

def get_info():
    return {
        "name": "Text to Speech",
        "description": "Converts text to speech using browser's built-in speech synthesis. Arguments: 'text' (required), 'voice' (optional: 'default', 'fast', 'slow'). The audio will play automatically when the response appears."
    }