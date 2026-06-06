import os
import requests
import time

# This grabs the magic key we saved in Railway
FAL_KEY = os.environ.get("FAL_KEY", "")

def run(args):
    """The main function that the AI Studio calls"""
    action = args.get("action", "3d").lower()
    prompt = args.get("prompt", "")
    
    if not prompt:
        return "❌ Error: Missing 'prompt'. Tell me what to create!"
    if not FAL_KEY:
        return "❌ FAL_KEY is missing in Railway variables!"

    if action == "video":
        return generate_video(prompt)
    else:
        return generate_3d(prompt)

def generate_3d(prompt):
    """Talks to the 3D factory (Using the newest Hunyuan3D-2 address)"""
    url = "https://queue.fal.run/fal-ai/hunyuan3d-2"
    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    data = {"prompt": prompt}

    try:
        response = requests.post(url, json=data, headers=headers)
        req_data = response.json()
        request_id = req_data.get("request_id")
        if not request_id: 
            return f"❌ 3D Factory rejected: {req_data}"

        status_url = f"https://queue.fal.run/fal-ai/hunyuan3d-2/requests/{request_id}/status"
        result_url = f"https://queue.fal.run/fal-ai/hunyuan3d-2/requests/{request_id}"
        
        while True:
            status_resp = requests.get(status_url, headers=headers).json()
            status = status_resp.get("status")
            if status == "COMPLETED":
                final = requests.get(result_url, headers=headers).json()
                # Find the 3D model link in the new response format
                model_url = (final.get("model_mesh_url") or 
                             final.get("model_url") or 
                             final.get("output", {}).get("model_mesh_url") or
                             final.get("output", {}).get("model_url"))
                
                if not model_url:
                    return f"❌ 3D Factory finished, but no model link found. Response: {final}"

                return f"✅ 3D Model Ready! <br><model-viewer src='{model_url}' auto-rotate camera-controls style='width:100%;height:400px;'></model-viewer>"
            elif status == "FAILED": 
                return "❌ 3D generation failed."
            time.sleep(3)
    except Exception as e:
        return f"❌ 3D Error: {str(e)}"

def generate_video(prompt):
    """Talks to the Hollywood Video factory (Minimax/Hailuo)"""
    url = "https://queue.fal.run/fal-ai/minimax/video-01"
    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    data = {"prompt": prompt}

    try:
        response = requests.post(url, json=data, headers=headers)
        req_data = response.json()
        request_id = req_data.get("request_id")
        if not request_id: 
            return f"❌ Video Factory rejected: {req_data}"

        status_url = f"https://queue.fal.run/fal-ai/minimax/video-01/requests/{request_id}/status"
        result_url = f"https://queue.fal.run/fal-ai/minimax/video-01/requests/{request_id}"
        
        while True:
            status_resp = requests.get(status_url, headers=headers).json()
            status = status_resp.get("status")
            if status == "COMPLETED":
                final = requests.get(result_url, headers=headers).json()
                video_url = final.get("video", {}).get("url") or final.get("output", {}).get("video", {}).get("url")
                return f"✅ Hyper-Realistic Video Ready! <br><video src='{video_url}' controls autoplay loop style='width:100%; max-height:500px; border-radius:10px;'></video>"
            elif status == "FAILED": 
                return "❌ Video generation failed."
            time.sleep(5)
    except Exception as e:
        return f"❌ Video Error: {str(e)}"

def get_info():
    return {
        "name": "God-Tier Media Studio",
        "description": "Creates interactive 3D models and hyper-realistic videos. Arguments: 'action' (must be '3d' or 'video'), 'prompt' (describe what to make)."
    }