#!/usr/bin/env python3
"""
Create a simple animated GIF showing the geDIG process
"""

from PIL import Image, ImageDraw, ImageFont
import math

def create_simple_animation():
    frames = []
    width, height = 800, 400
    
    # Define node positions
    nodes = {
        'Thermo': (150, 100),
        'Info': (650, 100),
        'Bio': (100, 300),
        'Physics': (400, 300),
        'Systems': (700, 300),
        'Entropy': (400, 200),  # Emerges later
    }
    
    # Frame 1: Initial state
    img1 = Image.new('RGB', (width, height), '#1e1e1e')
    draw1 = ImageDraw.Draw(img1)
    draw1.text((350, 20), "1. Initial Knowledge Graph", fill='white')
    
    # Draw initial nodes
    for node, (x, y) in nodes.items():
        if node != 'Entropy':
            draw1.ellipse([x-30, y-30, x+30, y+30], fill='#4ecdc4', outline='white')
            draw1.text((x-20, y-5), node[:4], fill='black')
    
    # Draw initial edges
    edges = [('Thermo', 'Physics'), ('Info', 'Systems'), ('Bio', 'Systems')]
    for n1, n2 in edges:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        draw1.line([x1, y1, x2, y2], fill='#666666', width=2)
    
    frames.append(img1)
    
    # Frame 2: Query injection
    img2 = img1.copy()
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([0, 0, width, 50], fill='#1e1e1e')
    draw2.text((250, 20), "2. Query: Entropy Relationship?", fill='yellow')
    
    # Highlight relevant nodes
    for node in ['Thermo', 'Info']:
        x, y = nodes[node]
        draw2.ellipse([x-35, y-35, x+35, y+35], fill='#ff6b6b', outline='yellow', width=3)
        draw2.text((x-20, y-5), node[:4], fill='white')
    
    frames.append(img2)
    
    # Frame 3: Message passing
    img3 = img2.copy()
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([0, 0, width, 50], fill='#1e1e1e')
    draw3.text((300, 20), "3. GNN Message Passing", fill='white')
    
    # Show message flow
    for node in ['Physics', 'Systems']:
        x, y = nodes[node]
        draw3.ellipse([x-40, y-40, x+40, y+40], fill='#ff9999', outline='white', width=2)
    
    # Draw message arrows
    draw3.text((250, 150), "Messages →", fill='yellow')
    draw3.text((500, 150), "← Messages", fill='yellow')
    
    frames.append(img3)
    
    # Frame 4: Entropy emerges
    img4 = img1.copy()
    draw4 = ImageDraw.Draw(img4)
    draw4.rectangle([0, 0, width, 50], fill='#1e1e1e')
    draw4.text((300, 20), "4. New Concept Emerges!", fill='#ffd93d')
    
    # Draw all original nodes
    for node, (x, y) in nodes.items():
        if node != 'Entropy':
            draw4.ellipse([x-30, y-30, x+30, y+30], fill='#4ecdc4', outline='white')
            draw4.text((x-20, y-5), node[:4], fill='black')
    
    # Draw ENTROPY node (new)
    x, y = nodes['Entropy']
    draw4.ellipse([x-40, y-40, x+40, y+40], fill='#ffd93d', outline='red', width=3)
    draw4.text((x-30, y-5), "ENTROPY", fill='black')
    
    frames.append(img4)
    
    # Frame 5: Insight spike!
    img5 = img4.copy()
    draw5 = ImageDraw.Draw(img5)
    draw5.rectangle([0, 0, width, 50], fill='#1e1e1e')
    draw5.text((250, 20), "5. ⚡ INSIGHT SPIKE! ⚡", fill='red')
    
    # Draw new connections
    new_edges = [('Thermo', 'Entropy'), ('Info', 'Entropy'), ('Physics', 'Entropy')]
    for n1, n2 in new_edges:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        draw5.line([x1, y1, x2, y2], fill='#ffd93d', width=4)
    
    # Metrics
    draw5.text((50, 350), "ΔGED: -0.92", fill='#ff6b6b')
    draw5.text((650, 350), "ΔIG: +0.56", fill='#ffd93d')
    
    frames.append(img5)
    
    # Frame 6: Final state
    img6 = img5.copy()
    draw6 = ImageDraw.Draw(img6)
    draw6.rectangle([0, 0, width, 50], fill='#1e1e1e')
    draw6.text((200, 20), "6. Knowledge Graph Restructured", fill='white')
    draw6.text((200, 370), "Thermodynamic & Information Entropy Unified!", fill='#4ecdc4')
    
    frames.append(img6)
    
    # Save as animated GIF
    output_path = "gedig_simple_animation.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[2000, 1500, 1500, 1500, 2000, 3000],  # ms per frame
        loop=0
    )
    
    print(f"✅ Simple animation created: {output_path}")
    
    # Verify it's animated
    img = Image.open(output_path)
    frame_count = 0
    try:
        while True:
            frame_count += 1
            img.seek(frame_count)
    except EOFError:
        pass
    
    print(f"🎬 Frames: {frame_count} (Animated: {'Yes' if frame_count > 1 else 'No'})")
    
    return output_path

if __name__ == "__main__":
    create_simple_animation()