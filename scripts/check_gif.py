from PIL import Image

# Check the GIF animation
import sys
gif_path = sys.argv[1] if len(sys.argv) > 1 else "gedig_animation.gif"
img = Image.open(gif_path)

print(f"Format: {img.format}")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")

# Check if it's animated
try:
    frame_count = 0
    durations = []
    while True:
        duration = img.info.get('duration', 0)
        durations.append(duration)
        frame_count += 1
        img.seek(frame_count)
except EOFError:
    pass

print(f"\nAnimation info:")
print(f"Total frames: {frame_count}")
print(f"Frame durations (ms): {durations}")
print(f"Is animated: {frame_count > 1}")
print(f"Total duration: {sum(durations)}ms ({sum(durations)/1000:.1f}s)")