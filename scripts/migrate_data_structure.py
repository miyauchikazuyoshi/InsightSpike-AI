#!/usr/bin/env python3
"""
Migrate data structure from snapshot to transaction approach
===========================================================

This script migrates the legacy data structure to the new SQLite-based structure.
"""

import os
import shutil
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import argparse
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class DataStructureMigrator:
    """Migrate from snapshot-based to transaction-based data structure"""
    
    def __init__(self, data_dir: str, dry_run: bool = False, auto_delete: bool = False):
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run
        self.auto_delete = auto_delete
        self.backup_dir = self.data_dir / f"migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def backup_current_structure(self):
        """Create full backup of current data directory"""
        if self.dry_run:
            print(f"[DRY RUN] Would create backup at: {self.backup_dir}")
            return
            
        print(f"Creating backup at: {self.backup_dir}")
        shutil.copytree(self.data_dir, self.backup_dir, dirs_exist_ok=True)
        print("✓ Backup completed")
        
    def create_new_structure(self):
        """Create new directory structure"""
        new_dirs = [
            self.data_dir / "sqlite",
            self.data_dir / "knowledge_base" / "initial", 
            self.data_dir / "knowledge_base" / "samples"
        ]
        
        for dir_path in new_dirs:
            if self.dry_run:
                print(f"[DRY RUN] Would create directory: {dir_path}")
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created: {dir_path}")
                
    def migrate_sample_data(self):
        """Move sample data to new location"""
        old_samples = self.data_dir / "samples"
        new_samples = self.data_dir / "knowledge_base" / "samples"
        
        if old_samples.exists():
            for file_path in old_samples.glob("*.txt"):
                new_path = new_samples / file_path.name
                if self.dry_run:
                    print(f"[DRY RUN] Would move: {file_path} → {new_path}")
                else:
                    shutil.move(str(file_path), str(new_path))
                    print(f"✓ Moved: {file_path.name}")
                    
    def migrate_raw_data(self):
        """Move raw data to knowledge_base/initial"""
        old_raw = self.data_dir / "raw"
        new_initial = self.data_dir / "knowledge_base" / "initial"
        
        if old_raw.exists():
            for file_path in old_raw.glob("*.txt"):
                new_path = new_initial / file_path.name
                if self.dry_run:
                    print(f"[DRY RUN] Would move: {file_path} → {new_path}")
                else:
                    shutil.move(str(file_path), str(new_path))
                    print(f"✓ Moved: {file_path.name}")
                    
    def create_migration_report(self):
        """Create a report of what will be deleted"""
        report_path = self.data_dir / "MIGRATION_REPORT.md"
        
        to_delete = [
            "core/",
            "db/", 
            "processed/",
            "experiments/",
            "backup/",
            "clean_backup/"
        ]
        
        report = f"""# Data Structure Migration Report
Generated: {datetime.now().isoformat()}

## Directories to be removed:
"""
        
        for dir_name in to_delete:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
                size_mb = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file()) / 1024 / 1024
                report += f"\n### {dir_name}\n"
                report += f"- Files: {file_count}\n"
                report += f"- Size: {size_mb:.1f} MB\n"
                report += f"- Status: {'Will be removed' if not self.dry_run else 'Would be removed (dry run)'}\n"
                
        report += f"\n## New structure created:\n"
        report += """
```
data/
├── sqlite/
├── knowledge_base/
│   ├── initial/
│   └── samples/
├── models/
├── logs/
└── cache/
```

## Backup location:
{backup_dir}
""".format(backup_dir=self.backup_dir if not self.dry_run else "[DRY RUN - no backup created]")
        
        if self.dry_run:
            print("\n" + "="*50)
            print(report)
            print("="*50)
        else:
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\n✓ Migration report saved to: {report_path}")
            
    def cleanup_old_structure(self):
        """Remove old directories (with confirmation)"""
        to_delete = [
            "core",
            "db", 
            "processed",
            "experiments",
            "backup",
            "clean_backup",
            "temp",
            "embedding",
            "learning"
        ]
        
        if self.dry_run:
            print("\n[DRY RUN] Would delete the following directories:")
            for dir_name in to_delete:
                dir_path = self.data_dir / dir_name
                if dir_path.exists():
                    print(f"  - {dir_name}/")
            return
            
        print("\n⚠️  The following directories will be PERMANENTLY DELETED:")
        for dir_name in to_delete:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                print(f"  - {dir_name}/")
                
        if self.auto_delete:
            print("\n[AUTO-DELETE] Deleting directories...")
            confirm = "DELETE"
        else:
            confirm = input("\nType 'DELETE' to confirm deletion: ")
            
        if confirm == "DELETE":
            for dir_name in to_delete:
                dir_path = self.data_dir / dir_name  
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    print(f"✓ Deleted: {dir_name}/")
        else:
            print("Deletion cancelled.")
            
    def update_gitignore(self):
        """Update .gitignore for new structure"""
        gitignore_path = self.data_dir / ".gitignore"
        
        new_gitignore = """# SQLite database files
sqlite/*.db
sqlite/*.db-wal
sqlite/*.db-shm

# Logs
logs/

# Cache
cache/

# Backup files
migration_backup_*/
*.backup
*.bak

# OS files
.DS_Store
"""
        
        if self.dry_run:
            print("\n[DRY RUN] Would update .gitignore")
        else:
            with open(gitignore_path, 'w') as f:
                f.write(new_gitignore)
            print("\n✓ Updated .gitignore")
            
    def run_migration(self):
        """Execute the full migration"""
        print(f"\n{'='*50}")
        print(f"Data Structure Migration {'(DRY RUN)' if self.dry_run else ''}")
        print(f"{'='*50}\n")
        
        # Step 1: Backup
        if not self.dry_run:
            self.backup_current_structure()
            
        # Step 2: Create new structure
        self.create_new_structure()
        
        # Step 3: Migrate data
        self.migrate_sample_data()
        self.migrate_raw_data()
        
        # Step 4: Create report
        self.create_migration_report()
        
        # Step 5: Update gitignore
        self.update_gitignore()
        
        # Step 6: Cleanup (if not dry run)
        if not self.dry_run:
            self.cleanup_old_structure()
            
        print(f"\n{'='*50}")
        print(f"Migration {'simulation' if self.dry_run else ''} completed!")
        if self.dry_run:
            print("\nRun without --dry-run to execute the actual migration.")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Migrate data structure from snapshot to transaction approach")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to data directory (default: ./data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making changes"
    )
    parser.add_argument(
        "--auto-delete",
        action="store_true",
        help="Automatically delete old directories without confirmation"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
        
    # Run migration
    migrator = DataStructureMigrator(args.data_dir, args.dry_run, args.auto_delete)
    migrator.run_migration()


if __name__ == "__main__":
    main()