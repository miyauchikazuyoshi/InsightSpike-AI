#!/usr/bin/env python3
"""
Config Migration Script
======================

Automatically migrates test files from using the legacy config system to the new
InsightSpikeConfig system.

Features:
- Finds all test files that import from legacy_config
- Updates imports to use the new config system
- Updates config instantiation patterns
- Creates a detailed report of changes made
- Supports dry-run mode to preview changes
- Creates backups of modified files
"""

import argparse
import ast
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Migration patterns for imports
IMPORT_PATTERNS = [
    # Direct imports
    (r'from\s+insightspike\.config\.legacy_config\s+import\s+Config',
     'from insightspike.config import InsightSpikeConfig'),
    (r'from\s+insightspike\.config\.legacy_config\s+import\s+get_config',
     'from insightspike.config import get_config'),
    (r'from\s+insightspike\.config\.legacy_config\s+import\s+(\w+Config)',
     r'from insightspike.config.models import \1'),
    # Import everything
    (r'from\s+insightspike\.config\.legacy_config\s+import\s+\*',
     'from insightspike.config import InsightSpikeConfig, get_config\nfrom insightspike.config.models import *'),
    # Module import
    (r'import\s+insightspike\.config\.legacy_config\s+as\s+(\w+)',
     r'import insightspike.config as \1'),
    (r'from\s+insightspike\.config\s+import\s+legacy_config',
     'from insightspike import config'),
]

# Config instantiation patterns
CONFIG_PATTERNS = [
    # Direct instantiation
    (r'config\s*=\s*Config\(\s*\)', 'config = InsightSpikeConfig()'),
    (r'cfg\s*=\s*Config\(\s*\)', 'cfg = InsightSpikeConfig()'),
    # Get config function
    (r'config\s*=\s*get_config\(\s*\)', 'config = get_config()'),
    (r'cfg\s*=\s*get_config\(\s*\)', 'cfg = get_config()'),
]

# Attribute access patterns that need updating
ATTRIBUTE_PATTERNS = [
    # Nested config access patterns
    (r'config\.embedding\.', 'config.core.'),
    (r'config\.llm\.', 'config.core.'),
    (r'cfg\.embedding\.', 'cfg.core.'),
    (r'cfg\.llm\.', 'cfg.core.'),
    # Specific attribute mappings
    (r'config\.llm\.model_name', 'config.core.llm_model'),
    (r'config\.llm\.provider', 'config.core.llm_provider'),
    (r'config\.embedding\.model_name', 'config.core.model_name'),
    (r'cfg\.llm\.model_name', 'cfg.core.llm_model'),
    (r'cfg\.llm\.provider', 'cfg.core.llm_provider'),
    (r'cfg\.embedding\.model_name', 'cfg.core.model_name'),
]


class ConfigMigrator:
    """Handles migration of config usage in Python files"""
    
    def __init__(self, root_dir: Path, dry_run: bool = False, backup: bool = True):
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.backup = backup
        self.report = {
            'files_found': [],
            'files_modified': [],
            'files_skipped': [],
            'errors': [],
            'changes': {}
        }
        
    def find_files_with_legacy_config(self) -> List[Path]:
        """Find all Python files that import from legacy_config"""
        files = []
        
        # Search patterns
        patterns = [
            "**/*.py",
            "tests/**/*.py",
            "experiments/**/*.py",
            "scripts/**/*.py"
        ]
        
        for pattern in patterns:
            for file_path in self.root_dir.glob(pattern):
                if file_path.is_file() and self._uses_legacy_config(file_path):
                    files.append(file_path)
                    
        # Remove duplicates
        files = list(set(files))
        self.report['files_found'] = [str(f.relative_to(self.root_dir)) for f in files]
        return files
    
    def _uses_legacy_config(self, file_path: Path) -> bool:
        """Check if a file imports from legacy_config"""
        try:
            content = file_path.read_text()
            return 'legacy_config' in content
        except Exception:
            return False
    
    def migrate_file(self, file_path: Path) -> Optional[str]:
        """Migrate a single file to use the new config system"""
        try:
            original_content = file_path.read_text()
            modified_content = original_content
            changes = []
            
            # Apply import patterns
            for pattern, replacement in IMPORT_PATTERNS:
                new_content, count = re.subn(pattern, replacement, modified_content, flags=re.MULTILINE)
                if count > 0:
                    changes.append(f"Updated import: {pattern} -> {replacement} ({count} occurrences)")
                    modified_content = new_content
            
            # Apply config instantiation patterns
            for pattern, replacement in CONFIG_PATTERNS:
                new_content, count = re.subn(pattern, replacement, modified_content)
                if count > 0:
                    changes.append(f"Updated config instantiation: {pattern} -> {replacement} ({count} occurrences)")
                    modified_content = new_content
            
            # Apply attribute access patterns
            for pattern, replacement in ATTRIBUTE_PATTERNS:
                new_content, count = re.subn(pattern, replacement, modified_content)
                if count > 0:
                    changes.append(f"Updated attribute access: {pattern} -> {replacement} ({count} occurrences)")
                    modified_content = new_content
            
            # Check if any changes were made
            if modified_content != original_content:
                relative_path = str(file_path.relative_to(self.root_dir))
                self.report['changes'][relative_path] = changes
                
                if not self.dry_run:
                    # Create backup
                    if self.backup:
                        backup_path = file_path.with_suffix(f'.bak.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                        shutil.copy2(file_path, backup_path)
                    
                    # Write modified content
                    file_path.write_text(modified_content)
                    self.report['files_modified'].append(relative_path)
                else:
                    self.report['files_modified'].append(relative_path + " (dry-run)")
                
                return modified_content
            else:
                relative_path = str(file_path.relative_to(self.root_dir))
                self.report['files_skipped'].append(relative_path + " (no changes needed)")
                return None
                
        except Exception as e:
            relative_path = str(file_path.relative_to(self.root_dir))
            self.report['errors'].append(f"{relative_path}: {str(e)}")
            return None
    
    def run_migration(self) -> Dict:
        """Run the full migration process"""
        print("🔍 Searching for files using legacy_config...")
        files = self.find_files_with_legacy_config()
        
        if not files:
            print("✅ No files found using legacy_config!")
            return self.report
        
        print(f"📄 Found {len(files)} files to migrate")
        
        for file_path in files:
            print(f"  Processing: {file_path.relative_to(self.root_dir)}")
            self.migrate_file(file_path)
        
        return self.report
    
    def generate_report(self) -> str:
        """Generate a detailed migration report"""
        lines = ["Migration Report", "=" * 80, ""]
        
        # Summary
        lines.append("Summary:")
        lines.append(f"  Files found: {len(self.report['files_found'])}")
        lines.append(f"  Files modified: {len(self.report['files_modified'])}")
        lines.append(f"  Files skipped: {len(self.report['files_skipped'])}")
        lines.append(f"  Errors: {len(self.report['errors'])}")
        lines.append("")
        
        # Files found
        if self.report['files_found']:
            lines.append("Files found with legacy_config:")
            for f in sorted(self.report['files_found']):
                lines.append(f"  - {f}")
            lines.append("")
        
        # Modified files with changes
        if self.report['files_modified']:
            lines.append("Files modified:")
            for f in sorted(self.report['files_modified']):
                lines.append(f"  - {f}")
                if f.replace(" (dry-run)", "") in self.report['changes']:
                    for change in self.report['changes'][f.replace(" (dry-run)", "")]:
                        lines.append(f"    • {change}")
            lines.append("")
        
        # Skipped files
        if self.report['files_skipped']:
            lines.append("Files skipped:")
            for f in sorted(self.report['files_skipped']):
                lines.append(f"  - {f}")
            lines.append("")
        
        # Errors
        if self.report['errors']:
            lines.append("Errors encountered:")
            for error in self.report['errors']:
                lines.append(f"  ❌ {error}")
            lines.append("")
        
        # Recommendations
        lines.append("Recommendations:")
        lines.append("  1. Review the changes in each modified file")
        lines.append("  2. Run your test suite to ensure everything still works")
        lines.append("  3. Check for any custom Config subclasses that may need manual migration")
        lines.append("  4. Update any documentation that references the old config system")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate test files from legacy_config to InsightSpikeConfig"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search for files (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save report to file (default: print to console)"
    )
    
    args = parser.parse_args()
    
    # Create migrator
    migrator = ConfigMigrator(
        root_dir=args.root,
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
    # Run migration
    print(f"🚀 Starting config migration{' (DRY RUN)' if args.dry_run else ''}...")
    print(f"📁 Root directory: {args.root}")
    print()
    
    report_data = migrator.run_migration()
    report_text = migrator.generate_report()
    
    # Output report
    print()
    print(report_text)
    
    if args.output:
        args.output.write_text(report_text)
        print(f"\n📝 Report saved to: {args.output}")
    
    # Exit code based on errors
    if report_data['errors']:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())