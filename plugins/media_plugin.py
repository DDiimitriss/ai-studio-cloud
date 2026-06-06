import os
import requests
import time

# This grabs the magic key we saved in Railway
FAL_KEY = os.environ.get("FAL_KEY", "")

def generate_3d(prompt):
    """The brain that talks to the 3D factory"""
    if not FAL_KEY:
        return "❌ FAL_KEY is missing in Railway variables!"

    # The address of the 3D factory (Tripo3D via Fal.ai)
    url = "https://queue.fal.run/fal-ai/tripo/draftv2"
    headers = {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt}

    try:
        # 1. Tell the factory to start building
        response = requests.post(url, json=data, headers=headers)
        result = response.json()
        request_id = result.get("request_id")
        
        if not request_id:
            return f"❌ Factory rejected the request: {result}"

        # 2. Wait and check if it's finished
        status_url = f"https://queue.fal.run/fal-ai/tripo/draftv2/requests/{request_id}/status"
        result_url = f"https://queue.fal.run/fal-ai/tripo/draftv2/requests/{request_id}"
        
        while True:
            status_resp = requests.get(status_url, headers=headers).json()
            status = status_resp.get("status")
            
            if status == "COMPLETED":
                # It's done! Go get the 3D model link
                final_result = requests.get(result_url, headers=headers).json()
                
                # Find the link to the 3D file
                model_url = final_result.get("model", {}).get("model_url")
                if not model_url:
                    model_url = final_result.get("output", {}).get("model_url")
                
                # Send back the special HTML tag that makes it interactive!
                return f"✅ 3D Model Ready! <br><model-viewer src='{model_url}' auto-rotate camera-controls style='width:100%;height:400px;'></model-viewer>"
                
            elif status == "FAILED":
                return "❌ 3D generation failed at the factory."
                
            time.sleep(3) # Wait 3 seconds before checking again

    except Exception as e:
        return f"❌ Error talking to factory: {str(e)}"

def get_info():
    # This tells your AI Studio what this plugin does
    return {
        "name": "3D Model Generator",
        "description": "Creates an interactive 3D model from text. Arguments: 'prompt' (describe the 3D object)."
    }