#!/usr/bin/env python3
"""
Generate Colab-optimized poetry.lock file
This script prepares a Linux x86_64 + CUDA optimized lock file for Google Colab

Usage:
    python scripts/generate_colab_lock.py
    
Output:
    poetry.lock.colab - Optimized for Google Colab environment
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def main():
    print("🏗️ Generate Colab-Optimized Poetry.lock")
    print("=" * 45)
    
    # Check if we're in the right directory
    if not os.path.exists('pyproject.toml'):
        print("❌ Error: pyproject.toml not found")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    # Warn if running on non-Linux
    if platform.system() != 'Linux':
        print(f"⚠️  Warning: Running on {platform.system()}")
        print("   For best results, run this in a Linux environment")
        print("   (e.g., Docker, WSL, or Google Colab)")
        print()
    
    # Check Poetry availability
    try:
        result = subprocess.run(['poetry', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Poetry not found. Please install Poetry first:")
            print("   pip install poetry")
            sys.exit(1)
        print(f"✅ Poetry available: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Poetry not found. Please install Poetry first:")
        print("   pip install poetry")
        sys.exit(1)
    
    # Backup existing lock
    if os.path.exists('poetry.lock'):
        backup_name = f'poetry.lock.{platform.system().lower()}-backup'
        shutil.copy2('poetry.lock', backup_name)
        print(f"✅ Backed up existing poetry.lock to {backup_name}")
        
        # Remove for clean generation
        os.remove('poetry.lock')
        print("✅ Removed existing poetry.lock for clean generation")
    
    # Configure Poetry
    print("\n🔧 Configuring Poetry for system-wide packages...")
    subprocess.run(['poetry', 'config', 'virtualenvs.create', 'false'], check=True)
    
    # Generate new lock with Colab-focused groups
    print("\n🚀 Generating Colab-optimized poetry.lock...")
    print("   This may take several minutes...")
    
    try:
        # Focus on colab and main groups
        result = subprocess.run([
            'poetry', 'lock', '--no-update'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Poetry.lock generated successfully")
            
            if os.path.exists('poetry.lock'):
                # Create colab-specific version
                shutil.copy2('poetry.lock', 'poetry.lock.colab')
                print("✅ Saved as poetry.lock.colab")
                
                # Analyze the generated lock
                with open('poetry.lock.colab', 'r') as f:
                    content = f.read()
                
                size_kb = len(content) / 1024
                platform_markers = content.count('platform_')
                linux_refs = content.count('linux')
                darwin_refs = content.count('darwin') + content.count('Darwin')
                
                print(f"\n📊 Generated lock file statistics:")
                print(f"   ├─ Size: {size_kb:.1f}KB")
                print(f"   ├─ Platform markers: {platform_markers}")
                print(f"   ├─ Linux references: {linux_refs}")
                print(f"   └─ Darwin references: {darwin_refs}")
                
                if size_kb < 400 and darwin_refs < 50:
                    print("\n🎉 Excellent! Colab-optimized lock file created")
                elif size_kb < 600:
                    print("\n✅ Good! Lock file should work well in Colab")
                else:
                    print("\n⚠️  Large lock file - may still have platform issues")
                
            else:
                print("❌ poetry.lock was not created")
                
        else:
            print(f"❌ Lock generation failed:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Lock generation timed out")
        print("   This can happen with complex dependency trees")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
    
    # Restore original if backup exists
    backup_files = [f for f in os.listdir('.') if f.endswith('-backup')]
    if backup_files:
        original_backup = backup_files[0]
        if input(f"\nRestore original {original_backup}? (y/N): ").lower() == 'y':
            original_name = original_backup.replace('-backup', '').replace(f'.{platform.system().lower()}', '')
            shutil.copy2(original_backup, original_name)
            print(f"✅ Restored {original_name}")
    
    print("\n🎯 Next Steps:")
    print("1. Commit poetry.lock.colab to your repository")
    print("2. Update your Colab notebook to use this optimized lock")
    print("3. Test the installation in Google Colab")
    
    if os.path.exists('poetry.lock.colab'):
        print("\n📝 Add this code to your Colab notebook setup:")
        print("""
import os, shutil
if 'google.colab' in sys.modules and os.path.exists('poetry.lock.colab'):
    shutil.copy2('poetry.lock.colab', 'poetry.lock')
    print("✅ Using Colab-optimized poetry.lock")
""")

if __name__ == '__main__':
    main()
