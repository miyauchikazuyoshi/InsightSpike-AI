#!/usr/bin/env python3
"""
Validate Mermaid diagram files for GitHub compatibility.

This script checks for common issues that prevent Mermaid diagrams
from rendering correctly on GitHub.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_mermaid_file(filepath: Path) -> List[Tuple[int, str]]:
    """Check a single Mermaid file for potential issues."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        # Check for template variables
        if '{{' in line or '}}' in line:
            issues.append((i, f"Template variable found: {line.strip()}"))
        
        # Check for problematic characters in node definitions
        if re.search(r'\[.*[<>].*\]', line):
            issues.append((i, f"HTML-like characters in node: {line.strip()}"))
        
        # Check for unescaped quotes in node text
        if re.search(r'\[.*["\'].*\]', line):
            issues.append((i, f"Unescaped quotes in node: {line.strip()}"))
        
        # Check for very long lines (GitHub has rendering limits)
        if len(line) > 200:
            issues.append((i, f"Line too long ({len(line)} chars)"))
    
    # Check file starts with valid diagram type
    if lines:
        first_non_comment = ''
        for l in lines:
            if l.strip() and not l.strip().startswith('%%'):
                first_non_comment = l
                break
        
        valid_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 
                      'stateDiagram', 'erDiagram', 'gantt', 'pie', 'gitGraph']
        
        if first_non_comment and not any(first_non_comment.strip().startswith(t) for t in valid_types):
            issues.append((1, f"No valid diagram type declaration found"))
    
    return issues


def main():
    """Main validation function."""
    diagrams_dir = Path(__file__).parent.parent.parent / 'docs' / 'diagrams'
    
    if not diagrams_dir.exists():
        print(f"Error: Diagrams directory not found: {diagrams_dir}")
        sys.exit(1)
    
    mermaid_files = list(diagrams_dir.glob('*.mermaid'))
    
    if not mermaid_files:
        print("No Mermaid files found.")
        return
    
    total_issues = 0
    
    for filepath in sorted(mermaid_files):
        issues = check_mermaid_file(filepath)
        
        if issues:
            print(f"\n❌ {filepath.name}")
            for line_num, issue in issues:
                print(f"   Line {line_num}: {issue}")
            total_issues += len(issues)
        else:
            print(f"✅ {filepath.name}")
    
    print(f"\n{'='*50}")
    print(f"Total files checked: {len(mermaid_files)}")
    print(f"Total issues found: {total_issues}")
    
    if total_issues > 0:
        print("\n⚠️  Fix these issues to ensure GitHub compatibility.")
        sys.exit(1)
    else:
        print("\n✨ All Mermaid files are valid!")


if __name__ == '__main__':
    main()
