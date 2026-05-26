import os
import time

def run(args):
    # Get folder – handle common names
    folder_input = args.get("folder", "Desktop")
    days_old = int(args.get("days_old", 7))
    action = args.get("action", "suggest")

    # Convert folder shortcuts to full path
    user_home = os.environ["USERPROFILE"]
    if folder_input.lower() == "desktop":
        folder = os.path.join(user_home, "Desktop")
    elif folder_input.lower() == "downloads":
        folder = os.path.join(user_home, "Downloads")
    elif folder_input.lower() == "documents":
        folder = os.path.join(user_home, "Documents")
    elif folder_input.lower() == "pictures":
        folder = os.path.join(user_home, "Pictures")
    elif folder_input.lower() == "videos":
        folder = os.path.join(user_home, "Videos")
    elif folder_input.lower() == "music":
        folder = os.path.join(user_home, "Music")
    else:
        folder = folder_input  # assume full path

    # Verify folder exists
    if not os.path.exists(folder):
        return f"Error: Folder '{folder_input}' does not exist. Tried path: {folder}"

    # Find files older than days_old
    now = time.time()
    old_files = []
    try:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                mod_time = os.path.getmtime(filepath)
                if now - mod_time > days_old * 86400:
                    old_files.append(filename)
    except Exception as e:
        return f"Error scanning folder: {e}"

    if not old_files:
        return f"No files older than {days_old} days found in '{folder_input}'."

    if action == "suggest":
        return f"Found {len(old_files)} old file(s) in '{folder_input}':\n- " + "\n- ".join(old_files[:10])
    elif action == "organize":
        old_folder = os.path.join(folder, "OldFiles")
        os.makedirs(old_folder, exist_ok=True)
        moved = []
        for fname in old_files:
            src = os.path.join(folder, fname)
            dst = os.path.join(old_folder, fname)
            os.rename(src, dst)
            moved.append(fname)
        return f"Moved {len(moved)} old files from '{folder_input}' to '{old_folder}'"
    else:
        return "Action must be 'suggest' or 'organize'."

def get_info():
    return {
        "name": "Proactive Agent",
        "description": "Scans a folder and suggests or moves old files. Arguments: 'folder' (Desktop, Downloads, or full path), 'days_old' (default 7), 'action' (suggest or organize). Use when user asks to clean up old files or see what hasn't been used recently."
    }