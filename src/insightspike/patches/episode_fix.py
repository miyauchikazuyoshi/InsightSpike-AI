"""
Fix for Episode parameter naming issue
"""

def fix_main_agent_episode_creation():
    """Fix Episode creation in MainAgent to use 'confidence' instead of 'c'."""
    import os
    from pathlib import Path
    
    # Find main_agent.py
    project_root = Path(__file__).parent.parent
    main_agent_path = project_root / "implementations" / "agents" / "main_agent.py"
    
    if not main_agent_path.exists():
        print(f"Warning: Could not find {main_agent_path}")
        return
    
    # Read the file
    content = main_agent_path.read_text()
    
    # Fix the Episode creation with c= parameter
    old_pattern = """                    episode = Episode(
                        text=text,
                        vec=embedding,
                        c=c_value,
                        timestamp=time.time(),
                        metadata={"c_value": c_value}
                    )"""
    
    new_pattern = """                    episode = Episode(
                        text=text,
                        vec=embedding,
                        confidence=c_value,
                        timestamp=time.time(),
                        metadata={"c_value": c_value}
                    )"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        main_agent_path.write_text(content)
        print("Fixed Episode creation in main_agent.py")
    else:
        print("Episode creation already fixed or pattern not found")
    
    # Also fix test files that use c= parameter
    test_files = [
        project_root.parent.parent / "tests" / "conftest.py",
        project_root.parent.parent / "tests" / "integration" / "test_main_agent_integration.py",
        project_root.parent.parent / "tests" / "unit" / "test_layer2_memory_manager.py"
    ]
    
    for test_file in test_files:
        if test_file.exists():
            content = test_file.read_text()
            # Replace Episode(..., c=value) with Episode(..., confidence=value)
            import re
            old_content = content
            content = re.sub(r'Episode\((.*?),\s*c=([^,\)]+)', r'Episode(\1, confidence=\2', content)
            if content != old_content:
                test_file.write_text(content)
                print(f"Fixed Episode creation in {test_file.name}")


def clean_corrupted_episodes():
    """Clean up corrupted episode files."""
    import os
    from pathlib import Path
    
    # Common locations for episode files
    home = Path.home()
    potential_paths = [
        home / "tmp" / "final_experiment" / "episodes.json",
        home / "tmp" / "quick_test" / "episodes.json",
        home / "tmp" / "debug_graph" / "episodes.json",
        home / ".insightspike" / "episodes" / "episodes.json",
    ]
    
    for path in potential_paths:
        if path.exists():
            try:
                # Try to read and parse the file
                import json
                with open(path, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                print(f"Found corrupted episodes file: {path}")
                print(f"Error: {e}")
                # Backup and remove
                backup_path = path.with_suffix('.json.corrupted')
                path.rename(backup_path)
                print(f"Moved to: {backup_path}")
            except Exception as e:
                print(f"Error checking {path}: {e}")


def fix_enum_value_error():
    """Fix str.value error by ensuring proper enum handling."""
    # This is likely happening in the LLM provider selection or similar enum usage
    # We need to identify where the error occurs
    pass  # Will implement after finding the exact location


if __name__ == "__main__":
    print("Applying Episode fixes...")
    fix_main_agent_episode_creation()
    clean_corrupted_episodes()
    print("Episode fixes applied!")