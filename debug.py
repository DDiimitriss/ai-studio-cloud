import sys
import traceback

try:
    with open('web_universal.py', 'r', encoding='utf-8') as f:
        code = f.read()
    exec(code)
except Exception as e:
    traceback.print_exc()
    input("\nPress Enter to exit...")