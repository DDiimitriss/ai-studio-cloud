import os
import requests
from moviepy.editor import ImageClip, concatenate_videoclips
from moviepy.video.fx import fadein, fadeout

def run(args):
    image_urls = args.get("image_urls", [])
    output_name = args.get("output_name", "generated_video")
    duration_per_image = int(args.get("duration", 3))
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")

    if not image_urls:
        return "Error: No image URLs provided."

    clips = []
    for idx, url in enumerate(image_urls):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                temp_path = os.path.join(desktop, f"temp_{idx}.jpg")
                with open(temp_path, "wb") as f:
                    f.write(resp.content)
                clip = ImageClip(temp_path).set_duration(duration_per_image)
                clip = clip.fx(fadein, 0.5).fx(fadeout, 0.5)
                clips.append(clip)
        except Exception as e:
            return f"Error downloading {url}: {e}"

    if not clips:
        return "No valid images downloaded."

    final_clip = concatenate_videoclips(clips, method="compose")
    output_path = os.path.join(desktop, f"{output_name}.mp4")
    final_clip.write_videofile(output_path, fps=24, logger=None, verbose=False)
    # Cleanup temp files
    for i in range(len(image_urls)):
        temp_path = os.path.join(desktop, f"temp_{i}.jpg")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return f"Video saved to {output_path}"

def get_info():
    return {
        "name": "Video Generator",
        "description": "Creates a slideshow video from image URLs. Arguments: 'image_urls' (list of image URLs), 'output_name' (name without .mp4), 'duration' (seconds per image, default 3). Use when user wants to create a video from pictures."
    }