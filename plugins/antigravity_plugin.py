import subprocess
import time

def run(args):
    task = args.get("task", "")
    if not task:
        return "Error: No task provided."

    print(f"[Antigravity] Running: agy --print {task[:50]}...")

    try:
        start = time.time()
        result = subprocess.run(
            ["agy", "--print", task],
            capture_output=True,
            text=True,
            timeout=120,
            shell=True
        )
        elapsed = time.time() - start
        print(f"[Antigravity] return code: {result.returncode}")
        print(f"[Antigravity] stdout length: {len(result.stdout)} chars")
        print(f"[Antigravity] stderr length: {len(result.stderr)} chars")
        print(f"[Antigravity] stdout first 200 chars: {result.stdout[:200]}")
        print(f"[Antigravity] stderr first 200 chars: {result.stderr[:200]}")

        if result.returncode == 0:
            output = result.stdout.strip()
            if not output:
                # If stdout empty, maybe output went to stderr? Use stderr instead
                output = result.stderr.strip()
            if not output:
                output = "(No output from Antigravity)"
            return f"Antigravity result ({elapsed:.1f}s):\n{output}"
        else:
            return f"Antigravity error (code {result.returncode}):\n{result.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return "Antigravity task timed out after 120 seconds."
    except FileNotFoundError:
        return "Antigravity CLI not found. Ensure 'agy' is in your PATH."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def get_info():
    return {
        "name": "Antigravity Agent",
        "description": "Delegates a task to Antigravity CLI using print mode. Arguments: 'task' (string)."
    }