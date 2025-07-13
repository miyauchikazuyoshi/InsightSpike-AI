#!/usr/bin/env python3
"""
Create an animation showing query transformation through message passing
The query itself evolves into an insightful answer
"""

from PIL import Image, ImageDraw, ImageFont
import math

def create_query_transformation_animation():
    frames = []
    width, height = 900, 500
    
    # Node positions
    nodes = {
        'Thermo': (150, 150),
        'Info': (750, 150),
        'Physics': (150, 350),
        'Systems': (750, 350),
        'Biology': (450, 100),
        'Energy': (300, 250),  # Hidden initially
        'Entropy': (600, 250), # Hidden initially
    }
    
    # Frame 1: Query placement evaluation
    img1 = Image.new('RGB', (width, height), '#1e1e1e')
    draw1 = ImageDraw.Draw(img1)
    draw1.text((300, 20), "1. Query Placement & geDIG Evaluation", fill='white')
    
    # Draw nodes
    for node, (x, y) in nodes.items():
        if node not in ['Energy', 'Entropy']:
            draw1.ellipse([x-25, y-25, x+25, y+25], fill='#4ecdc4', outline='white')
            draw1.text((x-20, y-5), node[:5], fill='black', font=ImageFont.load_default())
    
    # Draw query as a special node (tentative placement)
    query_x, query_y = 450, 250
    draw1.ellipse([query_x-40, query_y-40, query_x+40, query_y+40], 
                  fill='#ffff99', outline='yellow', width=3)
    draw1.text((query_x-35, query_y-15), "QUERY:", fill='black')
    draw1.text((query_x-35, query_y), "Entropy?", fill='black')
    
    # Show potential connections (dotted)
    for node in ['Thermo', 'Info']:
        x, y = nodes[node]
        # Draw dotted line
        for i in range(0, 10):
            x1 = query_x + (x - query_x) * i / 10
            y1 = query_y + (y - query_y) * i / 10
            x2 = query_x + (x - query_x) * (i + 0.5) / 10
            y2 = query_y + (y - query_y) * (i + 0.5) / 10
            draw1.line([x1, y1, x2, y2], fill='#666666', width=1)
    
    # geDIG metrics
    draw1.text((50, 450), "Evaluating: ΔGED potential = -0.3", fill='yellow')
    
    frames.append(img1)
    
    # Frame 2: Message passing begins
    img2 = Image.new('RGB', (width, height), '#1e1e1e')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((250, 20), "2. Message Passing Round 1: Query Absorbs", fill='white')
    
    # Redraw nodes
    for node, (x, y) in nodes.items():
        if node not in ['Energy', 'Entropy']:
            color = '#ff9999' if node in ['Thermo', 'Info'] else '#4ecdc4'
            draw2.ellipse([x-25, y-25, x+25, y+25], fill=color, outline='white')
            draw2.text((x-20, y-5), node[:5], fill='black')
    
    # Query starts transforming
    draw2.ellipse([query_x-45, query_y-45, query_x+45, query_y+45], 
                  fill='#ffcc66', outline='orange', width=3)
    draw2.text((query_x-40, query_y-20), "QUERY +", fill='black')
    draw2.text((query_x-40, query_y-5), "Thermo", fill='red')
    draw2.text((query_x-40, query_y+10), "Info", fill='red')
    
    # Show message flow
    for node in ['Thermo', 'Info']:
        x, y = nodes[node]
        draw2.line([x, y, query_x, query_y], fill='#ff6666', width=3)
        # Arrow
        angle = math.atan2(query_y - y, query_x - x)
        arrow_x = query_x - 50 * math.cos(angle)
        arrow_y = query_y - 50 * math.sin(angle)
        draw2.text((arrow_x-10, arrow_y-10), "→", fill='red')
    
    frames.append(img2)
    
    # Frame 3: Hidden connections emerge
    img3 = Image.new('RGB', (width, height), '#1e1e1e')
    draw3 = ImageDraw.Draw(img3)
    draw3.text((200, 20), "3. Hidden Connections Activate: Energy-Entropy", fill='white')
    
    # Draw all nodes including hidden ones
    for node, (x, y) in nodes.items():
        if node in ['Energy', 'Entropy']:
            draw3.ellipse([x-30, y-30, x+30, y+30], fill='#ffd93d', outline='yellow', width=2)
            draw3.text((x-25, y-5), node, fill='black')
        else:
            draw3.ellipse([x-25, y-25, x+25, y+25], fill='#4ecdc4', outline='white')
            draw3.text((x-20, y-5), node[:5], fill='black')
    
    # Query continues evolving
    draw3.ellipse([query_x-50, query_y-50, query_x+50, query_y+50], 
                  fill='#ff9966', outline='red', width=3)
    draw3.text((query_x-45, query_y-25), "EVOLVING", fill='white')
    draw3.text((query_x-45, query_y-10), "Energy ↔", fill='yellow')
    draw3.text((query_x-45, query_y+5), "Entropy", fill='yellow')
    draw3.text((query_x-45, query_y+20), "link?", fill='white')
    
    # New connections
    draw3.line([nodes['Energy'][0], nodes['Energy'][1], 
                nodes['Entropy'][0], nodes['Entropy'][1]], fill='#ffd93d', width=3)
    
    frames.append(img3)
    
    # Frame 4: Insight spike
    img4 = Image.new('RGB', (width, height), '#1e1e1e')
    draw4 = ImageDraw.Draw(img4)
    draw4.text((250, 20), "4. ⚡ INSIGHT SPIKE! Query Transforms ⚡", fill='red')
    
    # Draw complete graph
    for node, (x, y) in nodes.items():
        draw4.ellipse([x-25, y-25, x+25, y+25], fill='#4ecdc4', outline='white')
        draw4.text((x-20, y-5), node[:5], fill='black')
    
    # Draw all connections
    connections = [
        ('Thermo', 'Energy'), ('Energy', 'Entropy'), ('Entropy', 'Info'),
        ('Physics', 'Energy'), ('Systems', 'Info'), ('Thermo', 'Entropy')
    ]
    for n1, n2 in connections:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        draw4.line([x1, y1, x2, y2], fill='#ffd93d', width=2)
    
    # Query becomes insight
    draw4.ellipse([query_x-60, query_y-60, query_x+60, query_y+60], 
                  fill='#00ff00', outline='white', width=4)
    draw4.text((query_x-55, query_y-30), "INSIGHT:", fill='black', font=ImageFont.load_default())
    draw4.text((query_x-55, query_y-15), "Thermo &", fill='black')
    draw4.text((query_x-55, query_y), "Info entropy", fill='black')
    draw4.text((query_x-55, query_y+15), "= same math!", fill='black')
    draw4.text((query_x-55, query_y+30), "S = k log W", fill='red')
    
    # Metrics
    draw4.text((50, 450), "ΔGED: -0.92 (simplified!)", fill='#ff6b6b')
    draw4.text((650, 450), "ΔIG: +0.56 (unified!)", fill='#ffd93d')
    
    frames.append(img4)
    
    # Frame 5: Final answer
    img5 = Image.new('RGB', (width, height), '#1e1e1e')
    draw5 = ImageDraw.Draw(img5)
    draw5.text((200, 20), "5. Query → Answer: Complete Transformation", fill='white')
    
    # Show the complete graph faded
    for node, (x, y) in nodes.items():
        draw5.ellipse([x-20, y-20, x+20, y+20], fill='#2a4a4a', outline='#444444')
    
    # Show the answer prominently
    answer_box = [(200, 150), (700, 350)]
    draw5.rectangle(answer_box, fill='#003366', outline='#00ccff', width=3)
    draw5.text((220, 170), "ANSWER (transformed from query):", fill='#00ccff')
    draw5.text((220, 200), "Thermodynamic and information entropy are", fill='white')
    draw5.text((220, 220), "mathematically equivalent - both measure the", fill='white')
    draw5.text((220, 240), "number of possible microstates (W) of a system.", fill='white')
    draw5.text((220, 270), "S = k ln W (Boltzmann)", fill='#ffd93d')
    draw5.text((220, 290), "H = -Σ p log p (Shannon)", fill='#ffd93d')
    draw5.text((220, 320), "Information IS physical!", fill='#00ff00')
    
    frames.append(img5)
    
    # Frame 6: Summary
    img6 = Image.new('RGB', (width, height), '#1e1e1e')
    draw6 = ImageDraw.Draw(img6)
    draw6.text((250, 20), "6. geDIG Process Complete", fill='white')
    
    # Process diagram
    steps = [
        (150, 200, "Query", '#ffff99'),
        (300, 200, "→", 'white'),
        (450, 200, "Message\nPassing", '#ff9966'),
        (600, 200, "→", 'white'),
        (750, 200, "Insightful\nAnswer", '#00ff00')
    ]
    
    for x, y, text, color in steps:
        if text == "→":
            draw6.text((x, y), text, fill=color, font=ImageFont.load_default())
        else:
            draw6.ellipse([x-50, y-50, x+50, y+50], fill=color, outline='white')
            lines = text.split('\n')
            for i, line in enumerate(lines):
                draw6.text((x-40, y-10+i*15), line, fill='black')
    
    draw6.text((200, 350), "The query didn't just find an answer -", fill='#4ecdc4')
    draw6.text((200, 370), "it BECAME the answer through graph transformation!", fill='#4ecdc4')
    
    frames.append(img6)
    
    # Save as animated GIF
    output_path = "query_transformation_animation.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[2500, 2000, 2000, 2500, 2500, 3000],  # ms per frame
        loop=0
    )
    
    print(f"✅ Query transformation animation created: {output_path}")
    
    # Verify
    img = Image.open(output_path)
    frame_count = 0
    try:
        while True:
            frame_count += 1
            img.seek(frame_count)
    except EOFError:
        pass
    
    print(f"🎬 Frames: {frame_count} (Animated: {'Yes' if frame_count > 1 else 'No'})")
    import os
    print(f"📏 Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path

if __name__ == "__main__":
    create_query_transformation_animation()