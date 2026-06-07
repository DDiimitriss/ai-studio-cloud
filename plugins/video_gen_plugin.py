import os
from moviepy import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip

def get_info():
    return {
        "name": "Video Gen",
        "description": "Video editing plugin - concatenate clips, add text overlays, and process videos."
    }

def run(args):
    """Perform video operations using moviepy."""
    action = args.get("action", "info")
    
    try:
        if action == "info":
            return "✅ Video Gen plugin ready. Actions: concat, add_text, info"
        
        elif action == "concat":
            # Concatenate multiple video files
            video_files = args.get("videos", [])
            if not video_files:
                return "❌ No video files provided"
            
            clips = []
            for vf in video_files:
                if os.path.exists(vf):
                    clips.append(VideoFileClip(vf))
                else:
                    return f"❌ File not found: {vf}"
            
            if not clips:
                return "❌ No valid video files found"
            
            final = concatenate_videoclips(clips)
            output = args.get("output", "output_concat.mp4")
            final.write_videofile(output, codec="libx264", audio_codec="aac")
            
            # Clean up
            for clip in clips:
                clip.close()
            final.close()
            
            return f"✅ Concatenated {len(clips)} videos → {output}"
        
        elif action == "add_text":
            # Add text overlay to a video
            video_file = args.get("video")
            text = args.get("text", "Hello World")
            output = args.get("output", "output_text.mp4")
            
            if not video_file or not os.path.exists(video_file):
                return f"❌ Video file not found: {video_file}"
            
            video = VideoFileClip(video_file)
            
            # Create text clip
            txt_clip = TextClip(
                text=text,
                font_size=40,
                color='white',
                font="Arial",
                stroke_color='black',
                stroke_width=2
            )
            txt_clip = txt_clip.with_duration(video.duration).with_position('center')
            
            # Composite
            final = CompositeVideoClip([video, txt_clip])
            final.write_videofile(output, codec="libx264", audio_codec="aac")
            
            video.close()
            final.close()
            
            return f"✅ Added text overlay → {output}"
        
        else:
            return f"❌ Unknown action: {action}. Use 'concat', 'add_text', or 'info'."
    
    except Exception as e:
        return f"❌ Error: {str(e)}"