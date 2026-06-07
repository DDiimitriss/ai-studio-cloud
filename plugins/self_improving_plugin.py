import os
import json
import time

LOG_FILE = os.path.join(os.path.expanduser("~"), ".qwen_self_improvement.json")def load_lessons():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {"errors": [], "corrections": [], "lessons": []}

def save_lessons(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

def run(args):
    action = args.get("action", "recall")
    error_msg = args.get("error", "")
    correction = args.get("correction", "")
    lesson = args.get("lesson", "")

    data = load_lessons()

    if action == "log_error":
        data["errors"].append({"timestamp": time.time(), "error": error_msg})
        save_lessons(data)
        return f"Logged error: {error_msg[:100]}..."

    elif action == "log_correction":
        data["corrections"].append({"timestamp": time.time(), "correction": correction})
        save_lessons(data)
        return f"Logged correction: {correction[:100]}..."

    elif action == "log_lesson":
        data["lessons"].append({"timestamp": time.time(), "lesson": lesson})
        save_lessons(data)
        return f"Lesson learned: {lesson[:100]}..."

    elif action == "recall":
        output = "=== Past lessons ===\n"
        output += "\n".join([f"- {l['lesson'][:150]}" for l in data["lessons"][-5:]])
        output += "\n\n=== Past errors ===\n"
        output += "\n".join([f"- {e['error'][:150]}" for e in data["errors"][-5:]])
        output += "\n\n=== Corrections ===\n"
        output += "\n".join([f"- {c['correction'][:150]}" for c in data["corrections"][-5:]])
        return output

    else:
        return "Unknown action. Use 'log_error', 'log_correction', 'log_lesson', or 'recall'."

def get_info():
    return {
        "name": "Self Improving",
        "description": "Remembers mistakes and lessons to improve over time. Actions: log_error, log_correction, log_lesson, recall. Use when user wants to teach the AI or remember past mistakes."
    }