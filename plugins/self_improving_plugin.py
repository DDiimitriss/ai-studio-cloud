import os
import json
import time

# Universal path that works on Windows, Mac, and Linux
LOG_FILE = os.path.join(os.path.expanduser("~"), ".qwen_self_improvement.json")

def get_info():
    return {
        "name": "Self Improving",
        "description": "Tracks interactions and learns from user feedback to improve responses over time."
    }

def run(args):
    """Log an interaction or retrieve improvement data."""
    action = args.get("action", "log")
    
    try:
        # Load existing data
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"interactions": [], "feedback": [], "stats": {"total": 0}}
        
        if action == "log":
            # Log a new interaction
            interaction = {
                "timestamp": time.time(),
                "input": args.get("input", "")[:200],
                "output": args.get("output", "")[:200]
            }
            data["interactions"].append(interaction)
            data["stats"]["total"] += 1
            
            # Keep only last 100 interactions
            if len(data["interactions"]) > 100:
                data["interactions"] = data["interactions"][-100:]
            
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return f"✅ Logged interaction. Total: {data['stats']['total']}"
        
        elif action == "feedback":
            # Store user feedback
            feedback = {
                "timestamp": time.time(),
                "rating": args.get("rating", "neutral"),
                "comment": args.get("comment", "")[:200]
            }
            data["feedback"].append(feedback)
            
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return f"✅ Feedback recorded: {feedback['rating']}"
        
        elif action == "stats":
            # Return statistics
            return {
                "total_interactions": data["stats"]["total"],
                "feedback_count": len(data["feedback"]),
                "log_file": LOG_FILE
            }
        
        else:
            return f"❌ Unknown action: {action}. Use 'log', 'feedback', or 'stats'."
    
    except Exception as e:
        return f"❌ Error: {str(e)}"