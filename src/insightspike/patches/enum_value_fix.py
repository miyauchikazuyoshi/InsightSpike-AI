"""
Fix for Enum value access errors
"""

def fix_llm_interface_enum_access():
    """Fix L4LLMInterface to handle both Enum and string provider values."""
    import os
    from pathlib import Path
    
    # Find layer4_llm_interface.py
    project_root = Path(__file__).parent.parent
    llm_interface_path = project_root / "implementations" / "layers" / "layer4_llm_interface.py"
    
    if not llm_interface_path.exists():
        print(f"Warning: Could not find {llm_interface_path}")
        return
    
    # Read the file
    content = llm_interface_path.read_text()
    
    # Replace all instances of self.config.provider.value with provider value handling
    import re
    
    # Pattern to find self.config.provider.value
    pattern = r'self\.config\.provider\.value'
    
    # Replacement that handles both Enum and string
    replacement = '(self.config.provider.value if hasattr(self.config.provider, "value") else self.config.provider)'
    
    # Count replacements
    count = len(re.findall(pattern, content))
    
    if count > 0:
        content = re.sub(pattern, replacement, content)
        llm_interface_path.write_text(content)
        print(f"Fixed {count} instances of provider.value access in layer4_llm_interface.py")
    else:
        print("No provider.value access found or already fixed")
    
    # Also check for llm_config.provider.value
    pattern2 = r'llm_config\.provider\.value'
    replacement2 = '(llm_config.provider.value if hasattr(llm_config.provider, "value") else llm_config.provider)'
    
    count2 = len(re.findall(pattern2, content))
    
    if count2 > 0:
        content = re.sub(pattern2, replacement2, content)
        llm_interface_path.write_text(content)
        print(f"Fixed {count2} instances of llm_config.provider.value access")


def add_enum_value_fix_to_apply_fixes():
    """Add the enum fix to apply_fixes.py"""
    import os
    from pathlib import Path
    
    apply_fixes_path = Path(__file__).parent / "apply_fixes.py"
    
    if not apply_fixes_path.exists():
        print(f"Warning: Could not find {apply_fixes_path}")
        return
    
    content = apply_fixes_path.read_text()
    
    # Check if enum fix is already imported
    if "fix_llm_interface_enum_access" not in content:
        # Add import
        import_line = "from .enum_value_fix import fix_llm_interface_enum_access"
        content = content.replace(
            "from .l4_llm_fix import apply_l4_llm_fix",
            f"from .l4_llm_fix import apply_l4_llm_fix\n{import_line}"
        )
        
        # Add to apply_all_fixes
        content = content.replace(
            "    apply_l4_llm_fix()",
            "    apply_l4_llm_fix()\n    fix_llm_interface_enum_access()"
        )
        
        apply_fixes_path.write_text(content)
        print("Added enum fix to apply_fixes.py")


if __name__ == "__main__":
    print("Applying Enum value fixes...")
    fix_llm_interface_enum_access()
    add_enum_value_fix_to_apply_fixes()
    print("Enum value fixes applied!")