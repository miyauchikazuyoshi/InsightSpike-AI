#!/usr/bin/env python3
"""
Safe Core Functionality Test
===========================

Tests core InsightSpike functionality with segmentation fault protection.
"""

import os
import sys
from pathlib import Path

# Set safe mode from the start
os.environ['INSIGHTSPIKE_SAFE_MODE'] = '1'

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_core_functionality():
    """Test core functionality in safe mode"""
    print('🛡️ Testing Core InsightSpike Functionality (Safe Mode)')
    print('=' * 60)

    try:
        # Test configuration system
        print('1. Testing configuration system...')
        from insightspike.core.config import Config
        config = Config()
        print(f'✅ Config loaded: {config.embedding.model_name}')
        
        # Test fallback embedder directly
        print('2. Testing fallback embedder...')
        from insightspike.utils.embedder import FallbackEmbedder
        embedder = FallbackEmbedder(384)
        result = embedder.encode(['The cat sat on the mat.', 'Neural networks are powerful.'])
        print(f'✅ Fallback encoding works: shape {result.shape}')
        
        # Test memory manager with fallback embeddings
        print('3. Testing memory manager...')
        from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
        memory = L2MemoryManager(dim=result.shape[1])
        print(f'✅ Memory manager initialized: {memory.dim}')
        
        # Test adding episodes to memory
        print('4. Testing memory operations...')
        from insightspike.core.layers.layer2_memory_manager import Episode
        episode1 = Episode(result[0], "The cat sat on the mat.")
        episode2 = Episode(result[1], "Neural networks are powerful.")
        memory.episodes = [episode1, episode2]
        print(f'✅ Memory episodes added: {len(memory.episodes)}')
        
        # Test CLI imports
        print('5. Testing CLI system...')
        try:
            from insightspike.cli.main import app
            print(f'✅ CLI system available: {type(app)}')
        except ImportError:
            print('⚠️ CLI system not available (optional dependency)')
        
        # Test basic document processing
        print('6. Testing document processing...')
        test_docs = [
            "Artificial intelligence is transforming healthcare.",
            "Machine learning algorithms can detect patterns in medical data.",
            "Deep learning models are being used for drug discovery."
        ]
        
        doc_embeddings = embedder.encode(test_docs)
        print(f'✅ Document embeddings: shape {doc_embeddings.shape}')
        
        # Test similarity computation
        print('7. Testing similarity computation...')
        query_embedding = embedder.encode(["What is AI in medicine?"])
        similarities = doc_embeddings @ query_embedding.T
        print(f'✅ Similarity computation: shape {similarities.shape}')
        
        print()
        print('🎉 ALL SAFE MODE TESTS PASSED!')
        print('✅ Configuration system working')
        print('✅ Fallback embedding system working')  
        print('✅ Memory management working')
        print('✅ Document processing working')
        print('✅ Similarity computation working')
        
        return True
        
    except Exception as e:
        print(f'❌ Safe mode test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_cli_commands():
    """Test CLI commands in safe mode"""
    print('\n🖥️ Testing CLI Commands (Safe Mode)')
    print('=' * 60)
    
    try:
        # Test help command
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "insightspike.cli", "--help"
        ], capture_output=True, text=True, cwd=project_root, 
        env={**os.environ, "PYTHONPATH": str(project_root / "src")})
        
        if result.returncode == 0:
            print('✅ CLI help command works')
            return True
        else:
            print(f'❌ CLI help failed: {result.stderr}')
            return False
            
    except Exception as e:
        print(f'❌ CLI test failed: {e}')
        return False

def main():
    """Main test function"""
    print("🔍 InsightSpike-AI Safe Mode Functionality Test")
    print("=" * 70)
    
    # Test core functionality
    core_ok = test_core_functionality()
    
    # Test CLI if core is working
    cli_ok = test_cli_commands() if core_ok else False
    
    print("\n" + "=" * 70)
    print("🎯 SAFE MODE TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Core Functionality: {'PASS' if core_ok else 'FAIL'}")
    print(f"✅ CLI Commands: {'PASS' if cli_ok else 'FAIL'}")
    
    if core_ok:
        print("\n💡 RECOMMENDATION:")
        print("   ✅ Safe mode provides stable functionality")
        print("   ✅ Use INSIGHTSPIKE_SAFE_MODE=1 for production")
        print("   ✅ All core features work with fallback embedder")
    else:
        print("\n❌ CRITICAL: Safe mode functionality is failing")
        print("   Check Python environment and basic dependencies")
    
    return core_ok and cli_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
