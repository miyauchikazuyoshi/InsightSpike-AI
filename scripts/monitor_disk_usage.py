#!/usr/bin/env python3
"""
Monitor disk usage during experiment execution
"""

import os
import psutil
import time
import shutil
from datetime import datetime
from pathlib import Path
from insightspike.implementations.agents.main_agent import MainAgent

def get_directory_size(path):
    """Get total size of directory in MB"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total / (1024 * 1024)  # Convert to MB

def monitor_locations():
    """Monitor key locations for file creation"""
    locations = {
        "project_root": ".",
        "data_dir": "./data",
        "home_tmp": os.path.expanduser("~/tmp"),
        "home_library": os.path.expanduser("~/Library/Application Support"),
        "var_tmp": "/var/tmp",
        "var_folders": "/var/folders",
    }
    
    sizes = {}
    for name, path in locations.items():
        if os.path.exists(path):
            sizes[name] = get_directory_size(path)
        else:
            sizes[name] = 0
    
    return sizes

def find_new_files(before_files, after_files):
    """Find files created during execution"""
    new_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath not in before_files and filepath in after_files:
                try:
                    size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                    if size > 0.1:  # Only show files > 100KB
                        new_files.append((filepath, size))
                except:
                    pass
    return sorted(new_files, key=lambda x: x[1], reverse=True)

def get_all_files():
    """Get all files in project"""
    all_files = set()
    for root, dirs, files in os.walk("."):
        # Skip .git and .venv
        if ".git" in root or ".venv" in root:
            continue
        for file in files:
            all_files.add(os.path.join(root, file))
    return all_files

def main():
    print("=== Disk Usage Monitor ===")
    print(f"Start time: {datetime.now()}")
    
    # Initial measurements
    print("\n1. Initial disk usage:")
    disk_before = psutil.disk_usage("/").used / (1024**3)  # GB
    sizes_before = monitor_locations()
    files_before = get_all_files()
    
    for name, size in sizes_before.items():
        print(f"  {name}: {size:.2f} MB")
    print(f"  Total disk used: {disk_before:.2f} GB")
    
    # Run a simple experiment
    print("\n2. Running experiment...")
    config = {
        'llm': {'provider': 'mock'},
        'graph': {
            'enable_message_passing': True,
            'message_passing': {'alpha': 0.3, 'iterations': 2},
            'enable_graph_search': False,
            'use_gnn': False
        }
    }
    
    agent = MainAgent(config)
    
    # Add 10 knowledge items
    print("  Adding knowledge...")
    for i in range(10):
        agent.add_knowledge(f"Knowledge item {i}: This is a test of the emergency broadcast system.")
    
    # Process 10 questions
    print("  Processing questions...")
    for i in range(10):
        result = agent.process_question(f"What is knowledge item {i}?")
        print(f"    Processed question {i+1}/10")
    
    # Give time for any async writes
    time.sleep(2)
    
    # Final measurements
    print("\n3. Final disk usage:")
    disk_after = psutil.disk_usage("/").used / (1024**3)  # GB
    sizes_after = monitor_locations()
    files_after = get_all_files()
    
    for name, size in sizes_after.items():
        diff = size - sizes_before[name]
        if abs(diff) > 0.01:  # Only show changes > 10KB
            print(f"  {name}: {size:.2f} MB (+{diff:.2f} MB)")
        else:
            print(f"  {name}: {size:.2f} MB (no change)")
    
    print(f"  Total disk used: {disk_after:.2f} GB (+{disk_after-disk_before:.3f} GB)")
    
    # Find new files
    print("\n4. New files created:")
    new_files = find_new_files(files_before, files_after)
    if new_files:
        for filepath, size in new_files[:10]:  # Show top 10
            print(f"  {filepath}: {size:.2f} MB")
    else:
        print("  No new files > 100KB created")
    
    # Check for DataStore
    print("\n5. Checking DataStore locations:")
    datastore_paths = [
        "./data/insight_store",
        "./data/insightspike", 
        "./data/experiment_temp",
        os.path.expanduser("~/Library/Application Support/InsightSpike"),
        os.path.expanduser("~/.insightspike"),
    ]
    
    for path in datastore_paths:
        if os.path.exists(path):
            size = get_directory_size(path)
            print(f"  Found: {path} ({size:.2f} MB)")
            # List contents
            try:
                contents = os.listdir(path)
                for item in contents[:5]:  # Show first 5 items
                    item_path = os.path.join(path, item)
                    if os.path.isfile(item_path):
                        item_size = os.path.getsize(item_path) / (1024 * 1024)
                        print(f"    - {item}: {item_size:.2f} MB")
            except:
                pass
    
    print("\n=== Monitor Complete ===")

if __name__ == "__main__":
    main()