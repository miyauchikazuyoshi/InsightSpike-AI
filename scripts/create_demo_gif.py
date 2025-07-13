#!/usr/bin/env python3
"""
Create an animated GIF demo of InsightSpike-AI
"""

import sys
import time
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_frame(text, width=800, height=600, bg_color=(20, 20, 20), text_color=(255, 255, 255)):
    """Create a single frame with text"""
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a monospace font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", 14)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    y_offset = 20
    for line in text.split('\n'):
        draw.text((20, y_offset), line, fill=text_color, font=font)
        y_offset += 20
    
    return img

def create_demo_gif():
    """Create animated GIF showing InsightSpike in action"""
    frames = []
    
    # Frame 1: Title
    frame1_text = """
🧠 InsightSpike-AI Demo: Detecting 'Aha!' Moments

📚 Initializing InsightSpike system...
"""
    frames.append(create_frame(frame1_text))
    
    # Frame 2: System initialized
    frame2_text = """
🧠 InsightSpike-AI Demo: Detecting 'Aha!' Moments

📚 Initializing InsightSpike system...
✓ System initialized

📖 Adding knowledge to the system:
"""
    frames.append(create_frame(frame2_text))
    
    # Frame 3-7: Adding knowledge pieces
    knowledge_pieces = [
        "🌡️ Thermodynamics: Entropy always increases in isolated systems...",
        "💻 Information Theory: Information entropy measures uncertainty...",
        "🧬 Biology: Living systems maintain order by exporting entropy...",
        "⚡ Physics: Energy cannot be created or destroyed...",
        "🔄 Systems: Feedback loops can amplify or dampen changes..."
    ]
    
    base_text = """
🧠 InsightSpike-AI Demo: Detecting 'Aha!' Moments

📚 Initializing InsightSpike system...
✓ System initialized

📖 Adding knowledge to the system:
"""
    
    for i, knowledge in enumerate(knowledge_pieces):
        frame_text = base_text
        for j in range(i + 1):
            frame_text += f"  {knowledge_pieces[j]}\n"
        progress = "━" * ((i + 1) * 8) + " " * (40 - (i + 1) * 8)
        frame_text += f"\nLoading knowledge... {progress} {(i+1)*20}%"
        frames.append(create_frame(frame_text))
    
    # Frame 8: Knowledge ready
    frame8_text = base_text
    for k in knowledge_pieces:
        frame8_text += f"  {k}\n"
    frame8_text += """
Loading knowledge... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

✓ Knowledge base ready
"""
    frames.append(create_frame(frame8_text))
    
    # Frame 9: Question
    frame9_text = frame8_text + """
╭──────────────────────────────────────────────────────────────────╮
│ ❓ Question:                                                     │
│ How are thermodynamic entropy and information entropy related?   │
╰──────────────────────────────────────────────────────────────────╯

🔍 Processing question...
"""
    frames.append(create_frame(frame9_text))
    
    # Frame 10: Spike detected (with red highlight)
    spike_frame = create_frame(frame9_text, text_color=(255, 255, 255))
    draw = ImageDraw.Draw(spike_frame)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", 16)
    except:
        font = ImageFont.load_default()
    
    # Add spike detection in red
    spike_text = """

⚡ INSIGHT SPIKE DETECTED! ⚡
ΔGED: -0.920 (structure simplified)
ΔIG: 0.560 (information gained)
"""
    y_offset = 380
    for line in spike_text.split('\n'):
        if "INSIGHT SPIKE" in line:
            draw.text((20, y_offset), line, fill=(255, 100, 100), font=font)
        else:
            draw.text((20, y_offset), line, fill=(255, 255, 150), font=font)
        y_offset += 20
    
    frames.append(spike_frame)
    
    # Frame 11: Insight generated
    final_text = """
🧠 InsightSpike-AI Demo: Detecting 'Aha!' Moments

⚡ INSIGHT SPIKE DETECTED! ⚡

💡 Novel Insight Generated:
╭────────────────────── Aha! Moment ───────────────────────╮
│ Thermodynamic and information entropy are mathematically │
│ equivalent - both measure the number of possible         │
│ microstates of a system. This deep connection reveals    │
│ that information processing requires energy!             │
╰──────────────────────────────────────────────────────────╯

📊 InsightSpike Metrics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Metric              ┃ Before ┃ After ┃ Change   ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ Graph Edit Distance │ 2.84   │ 1.92  │ -0.92 ✨ │
│ Information Entropy │ 3.21   │ 2.65  │ -0.56 📉 │
│ Knowledge Nodes     │ 5      │ 7     │ +2 🆕    │
└─────────────────────┴────────┴───────┴──────────┘

🎯 InsightSpike created new knowledge connections!
"""
    
    # Create final frame with green highlight for insight
    final_frame = create_frame("", bg_color=(20, 20, 20))
    draw = ImageDraw.Draw(final_frame)
    
    y_offset = 20
    for line in final_text.split('\n'):
        if "Novel Insight" in line or "Aha!" in line:
            color = (100, 255, 100)  # Green for insight
        elif "SPIKE DETECTED" in line:
            color = (255, 100, 100)  # Red for spike
        elif "━" in line or "┃" in line or "┏" in line:
            color = (200, 200, 200)  # Gray for table
        else:
            color = (255, 255, 255)  # White for normal text
        
        draw.text((20, y_offset), line, fill=color, font=font)
        y_offset += 16
    
    frames.append(final_frame)
    
    # Add pause frames at the end
    for _ in range(3):
        frames.append(final_frame)
    
    # Save as GIF
    output_path = Path(__file__).parent.parent / "demo.gif"
    
    # Create duration list matching number of frames
    durations = [1000] + [500] * 5 + [1000, 1000, 2000, 3000] + [3000] * (len(frames) - 10)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations[:len(frames)],  # ms per frame
        loop=0
    )
    
    print(f"✅ Demo GIF created: {output_path}")
    print(f"📏 Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"🖼️  Frames: {len(frames)}")
    
    return output_path

if __name__ == "__main__":
    create_demo_gif()