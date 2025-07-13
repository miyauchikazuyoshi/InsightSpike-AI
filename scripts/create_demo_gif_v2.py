#!/usr/bin/env python3
"""
Create an animated GIF demo of InsightSpike-AI with better Unicode support
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

def create_frame(text, width=800, height=600, bg_color=(20, 20, 20), text_color=(255, 255, 255)):
    """Create a single frame with text - simplified Unicode handling"""
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Use default font for better Unicode support
    font = ImageFont.load_default()
    
    # Draw text line by line
    y_offset = 20
    for line in text.split('\n'):
        # Replace problematic Unicode characters with ASCII alternatives
        line = line.replace('━', '-')
        line = line.replace('┃', '|')
        line = line.replace('┏', '+')
        line = line.replace('┓', '+')
        line = line.replace('┗', '+')
        line = line.replace('┛', '+')
        line = line.replace('┳', '+')
        line = line.replace('┻', '+')
        line = line.replace('┣', '+')
        line = line.replace('┫', '+')
        line = line.replace('╇', '+')
        line = line.replace('╰', '+')
        line = line.replace('╭', '+')
        line = line.replace('╮', '+')
        line = line.replace('─', '-')
        line = line.replace('│', '|')
        line = line.replace('╯', '+')
        line = line.replace('🧠', '[BRAIN]')
        line = line.replace('📚', '[BOOK]')
        line = line.replace('✓', '[OK]')
        line = line.replace('📖', '[READ]')
        line = line.replace('🌡️', '[THERMO]')
        line = line.replace('💻', '[COMP]')
        line = line.replace('🧬', '[BIO]')
        line = line.replace('⚡', '[ENERGY]')
        line = line.replace('🔄', '[CYCLE]')
        line = line.replace('❓', '[Q]')
        line = line.replace('🔍', '[SEARCH]')
        line = line.replace('💡', '[IDEA]')
        line = line.replace('📊', '[CHART]')
        line = line.replace('🎯', '[TARGET]')
        line = line.replace('✨', '*')
        line = line.replace('📉', 'v')
        line = line.replace('🆕', 'NEW')
        
        draw.text((20, y_offset), line, fill=text_color, font=font)
        y_offset += 15
    
    return img

def create_demo_gif_v2():
    """Create animated GIF with ASCII-friendly characters"""
    frames = []
    
    # Frame 1: Title
    frame1_text = """
[BRAIN] InsightSpike-AI Demo: Detecting 'Aha!' Moments

[BOOK] Initializing InsightSpike system...
"""
    frames.append(create_frame(frame1_text))
    
    # Frame 2: System initialized
    frame2_text = """
[BRAIN] InsightSpike-AI Demo: Detecting 'Aha!' Moments

[BOOK] Initializing InsightSpike system...
[OK] System initialized

[READ] Adding knowledge to the system:
"""
    frames.append(create_frame(frame2_text))
    
    # Frame 3-7: Adding knowledge
    knowledge_pieces = [
        "[THERMO] Thermodynamics: Entropy always increases...",
        "[COMP] Information Theory: Information entropy measures...",
        "[BIO] Biology: Living systems maintain order...",
        "[ENERGY] Physics: Energy cannot be created or destroyed...",
        "[CYCLE] Systems: Feedback loops can amplify changes..."
    ]
    
    base_text = """
[BRAIN] InsightSpike-AI Demo: Detecting 'Aha!' Moments

[BOOK] Initializing InsightSpike system...
[OK] System initialized

[READ] Adding knowledge to the system:
"""
    
    for i, knowledge in enumerate(knowledge_pieces):
        frame_text = base_text
        for j in range(i + 1):
            frame_text += f"  {knowledge_pieces[j]}\n"
        progress = "=" * ((i + 1) * 8) + " " * (40 - (i + 1) * 8)
        frame_text += f"\nLoading... [{progress}] {(i+1)*20}%"
        frames.append(create_frame(frame_text))
    
    # Frame 8: Knowledge ready
    frame8_text = base_text
    for k in knowledge_pieces:
        frame8_text += f"  {k}\n"
    frame8_text += """
Loading... [========================================] 100%

[OK] Knowledge base ready
"""
    frames.append(create_frame(frame8_text))
    
    # Frame 9: Question
    frame9_text = frame8_text + """
+------------------------------------------------------------------+
| [Q] Question:                                                    |
| How are thermodynamic entropy and information entropy related?   |
+------------------------------------------------------------------+

[SEARCH] Processing question...
"""
    frames.append(create_frame(frame9_text))
    
    # Frame 10: Spike detected
    spike_text = frame9_text + """

[ENERGY] INSIGHT SPIKE DETECTED! [ENERGY]
DELTA_GED: -0.920 (structure simplified)
DELTA_IG: 0.560 (information gained)
"""
    frames.append(create_frame(spike_text, text_color=(255, 200, 200)))
    
    # Frame 11: Final insight
    final_text = """
[BRAIN] InsightSpike-AI Demo: Detecting 'Aha!' Moments

[ENERGY] INSIGHT SPIKE DETECTED! [ENERGY]

[IDEA] Novel Insight Generated:
+----------------------- Aha! Moment -----------------------+
| Thermodynamic and information entropy are mathematically  |
| equivalent - both measure the number of possible          |
| microstates of a system. This deep connection reveals     |
| that information processing requires energy!              |
+----------------------------------------------------------+

[CHART] InsightSpike Metrics
+---------------------+--------+-------+----------+
| Metric              | Before | After | Change   |
+---------------------+--------+-------+----------+
| Graph Edit Distance | 2.84   | 1.92  | -0.92 *  |
| Information Entropy | 3.21   | 2.65  | -0.56 v  |
| Knowledge Nodes     | 5      | 7     | +2 NEW   |
+---------------------+--------+-------+----------+

[TARGET] InsightSpike created new knowledge connections!
"""
    frames.append(create_frame(final_text))
    
    # Add pause frames
    for _ in range(3):
        frames.append(create_frame(final_text))
    
    # Save as GIF
    output_path = Path(__file__).parent.parent / "demo_v2.gif"
    
    # Create duration list
    durations = [1000] + [500] * 5 + [1000, 1000, 2000, 3000] + [3000] * (len(frames) - 10)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations[:len(frames)],
        loop=0,
        optimize=True
    )
    
    print(f"✅ Demo GIF v2 created: {output_path}")
    print(f"📏 Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"🖼️  Frames: {len(frames)}")
    
    return output_path

if __name__ == "__main__":
    create_demo_gif_v2()