from PIL import Image, ImageDraw
import numpy as np

# Create simple test frames
frames = []
for i in range(5):
    # Create a new image
    img = Image.new('RGB', (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw frame number
    draw.text((50, 40), f"Frame {i+1}", fill=(0, 0, 0))
    draw.rectangle([10 + i*30, 10, 40 + i*30, 40], fill=(255, 0, 0))
    
    frames.append(img)

# Save as animated GIF
frames[0].save(
    'test_animation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=500,
    loop=0
)

print("Test GIF created!")

# Check the result
test_img = Image.open('test_animation.gif')
frame_count = 0
try:
    while True:
        frame_count += 1
        test_img.seek(frame_count)
except EOFError:
    pass

print(f"Frames in test GIF: {frame_count}")