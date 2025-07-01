#!/usr/bin/env python3
"""
Data Restoration Utility
========================

Utility script to restore clean data state from backup.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse

def restore_clean_data(backup_dir="data/clean_backup", target_dir="data", confirm=True):
    """Restore clean data from backup directory"""
    
    backup_path = Path(backup_dir)
    target_path = Path(target_dir)
    
    if not backup_path.exists():
        print(f"❌ Backup directory not found: {backup_path}")
        return False
    
    # Files to restore
    restore_files = {
        "episodes_clean.json": "episodes.json",
        "graph_pyg_clean.pt": "graph_pyg.pt", 
        "index_clean.faiss": "index.faiss",
        "insight_facts_clean.db": "insight_facts.db",
        "unknown_learning_clean.db": "unknown_learning.db"
    }
    
    print("🔄 Data Restoration Utility")
    print("=" * 40)
    
    # Check backup files
    missing_files = []
    for backup_file in restore_files.keys():
        if not (backup_path / backup_file).exists():
            missing_files.append(backup_file)
    
    if missing_files:
        print(f"❌ Missing backup files: {missing_files}")
        return False
    
    # Show current data status
    print("📊 Current Data Status:")
    for target_file in restore_files.values():
        target_filepath = target_path / target_file
        if target_filepath.exists():
            size = target_filepath.stat().st_size
            modified = datetime.fromtimestamp(target_filepath.stat().st_mtime)
            print(f"  {target_file}: {size:,} bytes (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"  {target_file}: Not found")
    
    # Show backup status
    print(f"\n📦 Available Clean Backup:")
    for backup_file in restore_files.keys():
        backup_filepath = backup_path / backup_file
        size = backup_filepath.stat().st_size
        modified = datetime.fromtimestamp(backup_filepath.stat().st_mtime)
        print(f"  {backup_file}: {size:,} bytes (created: {modified.strftime('%Y-%m-%d %H:%M')})")
    
    # Confirmation
    if confirm:
        print(f"\n⚠️  This will overwrite current data files!")
        response = input("Continue? (y/N): ").lower().strip()
        if response != 'y':
            print("❌ Restoration cancelled")
            return False
    
    # Perform restoration
    print(f"\n🔄 Restoring clean data...")
    success_count = 0
    
    for backup_file, target_file in restore_files.items():
        try:
            backup_filepath = backup_path / backup_file
            target_filepath = target_path / target_file
            
            shutil.copy2(backup_filepath, target_filepath)
            print(f"  ✅ {backup_file} → {target_file}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Failed to restore {backup_file}: {e}")
    
    print(f"\n🎉 Restoration completed: {success_count}/{len(restore_files)} files")
    
    if success_count == len(restore_files):
        print("✅ All data files successfully restored to clean state")
        return True
    else:
        print("⚠️ Some files failed to restore")
        return False

def create_backup(source_dir="data", backup_dir="data/clean_backup"):
    """Create a new clean backup from current data"""
    
    source_path = Path(source_dir)
    backup_path = Path(backup_dir)
    
    backup_path.mkdir(exist_ok=True)
    
    backup_files = {
        "episodes.json": "episodes_clean.json",
        "graph_pyg.pt": "graph_pyg_clean.pt",
        "index.faiss": "index_clean.faiss", 
        "insight_facts.db": "insight_facts_clean.db",
        "unknown_learning.db": "unknown_learning_clean.db"
    }
    
    print("💾 Creating Clean Data Backup")
    print("=" * 40)
    
    success_count = 0
    for source_file, backup_file in backup_files.items():
        try:
            source_filepath = source_path / source_file
            backup_filepath = backup_path / backup_file
            
            if source_filepath.exists():
                shutil.copy2(source_filepath, backup_filepath)
                print(f"  ✅ {source_file} → {backup_file}")
                success_count += 1
            else:
                print(f"  ⚠️ {source_file}: Not found")
                
        except Exception as e:
            print(f"  ❌ Failed to backup {source_file}: {e}")
    
    print(f"\n🎉 Backup completed: {success_count}/{len(backup_files)} files")
    return success_count == len(backup_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data restoration utility for InsightSpike-AI")
    parser.add_argument("--restore", action="store_true", help="Restore clean data from backup")
    parser.add_argument("--backup", action="store_true", help="Create new clean backup")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    if args.backup:
        create_backup()
    elif args.restore:
        restore_clean_data(confirm=not args.force)
    else:
        print("Usage: python restore_clean_data.py --restore|--backup [--force]")
        print("  --restore: Restore data from clean backup")
        print("  --backup:  Create new clean backup") 
        print("  --force:   Skip confirmation prompts")
