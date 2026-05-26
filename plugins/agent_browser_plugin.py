from playwright.sync_api import sync_playwright
import os
import traceback
from datetime import datetime

def run(args):
    url = args.get("url")
    action = args.get("action")
    selector = args.get("selector", "")
    value = args.get("value", "")

    if not url:
        return "Error: No URL provided."

    try:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate
            page.goto(url, wait_until="domcontentloaded", timeout=300000)
            page.wait_for_timeout(3000)

            if action == "screenshot":
                # Use AI_Studio folder (guaranteed writable)
                studio_folder = os.path.dirname(os.path.abspath(__file__))  # plugins folder
                # Go up one level to AI_Studio
                ai_studio_folder = os.path.dirname(studio_folder)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(ai_studio_folder, f"screenshot_{timestamp}.png")
                
                # Also try Desktop but with unique name
                desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
                desktop_screenshot = os.path.join(desktop, f"screenshot_{timestamp}.png")
                
                # Save to both locations for testing
                page.screenshot(path=screenshot_path)
                page.screenshot(path=desktop_screenshot)
                
                browser.close()
                
                # Verify files exist
                file1_exists = os.path.exists(screenshot_path)
                file2_exists = os.path.exists(desktop_screenshot)
                
                result = f"Screenshot saved to {screenshot_path} (exists: {file1_exists}) and {desktop_screenshot} (exists: {file2_exists})"
                return result
            elif action == "click":
                if not selector:
                    return "Error: No selector for click action."
                page.click(selector)
                browser.close()
                return f"Clicked element '{selector}' on {url}"
            elif action == "fill":
                if not selector or not value:
                    return "Error: Need both selector and value for fill action."
                page.fill(selector, value)
                browser.close()
                return f"Filled '{selector}' with '{value}' on {url}"
            elif action == "text":
                if not selector:
                    return "Error: No selector for text extraction."
                text = page.inner_text(selector)
                browser.close()
                return f"Text from '{selector}':\n{text[:500]}"
            else:
                browser.close()
                return "Action must be screenshot, click, fill, or text."
    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        return f"Browser automation error: {str(e)}"

def get_info():
    return {
        "name": "Agent Browser",
        "description": "Automates browser tasks: take screenshot, click element, fill form, extract text. Arguments: 'url', 'action' (screenshot/click/fill/text), 'selector' (CSS selector), 'value' (for fill). Use when user wants to interact with a webpage."
    }