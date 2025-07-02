#!/usr/bin/env python3
"""
Critical Data Restoration Script
================================

Restores critical experimental data from backup to the main data/ directory.
"""

import os
import shutil
import json
from pathlib import Path

def restore_critical_data():
    """Restore critical data from backup to main data directory"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    backup_dir = project_root / "experiments" / "data_preservation" / "critical_backup"
    data_dir = project_root / "data"
    
    print("🔄 InsightSpike-AI Critical Data Restoration")
    print("=" * 50)
    
    # Verify backup exists
    if not backup_dir.exists():
        print("❌ Critical backup directory not found!")
        print(f"   Expected: {backup_dir}")
        return False
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Load backup metadata
    metadata_path = backup_dir.parent / "backup_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📊 Backup Date: {metadata['backup_date']}")
        print(f"📦 Backup Size: {metadata['backup_size']}")
    
    # Restore files
    restored_files = []
    
    for item in backup_dir.iterdir():
        if item.is_file():
            # Copy individual files
            dest_path = data_dir / item.name
            shutil.copy2(item, dest_path)
            restored_files.append(item.name)
            print(f"✅ Restored: {item.name}")
            
        elif item.is_dir():
            # Copy directories
            dest_path = data_dir / item.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(item, dest_path)
            restored_files.append(f"{item.name}/")
            print(f"✅ Restored: {item.name}/ (directory)")
    
    # Verification
    print("\n🔍 Verification:")
    for file_name in restored_files:
        file_path = data_dir / file_name.rstrip('/')
        if file_path.exists():
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   ✅ {file_name} ({size:,} bytes)")
            else:
                print(f"   ✅ {file_name} (directory)")
        else:
            print(f"   ❌ {file_name} (missing)")
    
    print(f"\n🎉 Successfully restored {len(restored_files)} items to {data_dir}")
    
    # Next steps
    print("\n📋 Next Steps:")
    print("   1. Run download scripts to restore HuggingFace datasets")
    print("   2. Rebuild FAISS index if needed: python rebuild_index.py")
    print("   3. Run test suite to verify environment: pytest tests/")
    print("   4. Check InsightSpike-AI functionality: python -m insightspike.test")
    
    return True

def create_missing_directories():
    """Create standard data directory structure"""
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    standard_dirs = [
        "cache", "embedding", "logs", "models", 
        "processed", "raw", "samples"
    ]
    
    for dir_name in standard_dirs:
        dir_path = data_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        
        # Create .gitkeep files
        gitkeep_path = dir_path / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
    
    print("📁 Created standard data directory structure")

if __name__ == "__main__":
    print("Starting critical data restoration...")
    
    # Create directory structure
    create_missing_directories()
    
    # Restore critical data
    success = restore_critical_data()
    
    if success:
        print("\n✅ Critical data restoration completed successfully!")
    else:
        print("\n❌ Critical data restoration failed!")
        exit(1)