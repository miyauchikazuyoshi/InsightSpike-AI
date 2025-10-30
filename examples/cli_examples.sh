#!/bin/bash
# CLI Examples for InsightSpike
# ==============================

echo "=== InsightSpike CLI Examples ==="
echo

# Basic usage
echo "1. Ask a question:"
echo "spike ask 'What is the relationship between quantum computing and information theory?'"
echo

# Short alias
echo "2. Quick question (using alias):"
echo "spike q 'How does entropy relate to machine learning?'"
echo

# Learn from documents
echo "3. Learn from a file:"
echo "spike learn data/raw/knowledge.txt"
echo

echo "4. Learn from a directory:"
echo "spike learn data/raw/"
echo

# Configuration management
echo "5. Show current configuration:"
echo "spike config show"
echo

echo "6. Change settings:"
echo "spike config set safe_mode false"
echo "spike config set max_tokens 512"
echo

echo "7. Use configuration presets:"
echo "spike config preset experiment  # Real LLM, moderate settings"
echo "spike config preset production  # Optimized for performance"
echo

echo "8. Save/load configuration:"
echo "spike config save my_config.json"
echo "spike config load my_config.json"
echo

# Interactive mode
echo "9. Start interactive chat:"
echo "spike chat"
echo "# or use alias:"
echo "spike c"
echo

# Statistics
echo "10. Show agent statistics:"
echo "spike stats"
echo

# Experiments
echo "11. Run experiments:"
echo "spike experiment --name simple --episodes 10"
echo "spike experiment --name insight --episodes 5"
echo "spike experiment --name math --episodes 7"
echo

# Advanced usage
echo "12. Ask with specific preset:"
echo "spike ask 'Complex question' --preset experiment"
echo

echo "13. Verbose output:"
echo "spike ask 'Question' --verbose"
echo "# or:"
echo "spike ask 'Question' -v"
echo

# Version info
echo "14. Show version:"
echo "spike version"
echo

# Environment variables
echo "15. Using environment variables:"
echo "export INSIGHTSPIKE_SAFE_MODE=false"
echo "export INSIGHTSPIKE_MAX_TOKENS=1024"
echo "spike ask 'Question using env config'"
echo

# Chat mode commands
echo "16. Commands in chat mode:"
echo "# Type these after starting 'spike chat':"
echo "#   help     - Show available commands"
echo "#   stats    - Show statistics"
echo "#   config   - Show configuration"
echo "#   clear    - Clear conversation history"
echo "#   exit     - Exit chat mode"
echo

echo "=== Tips ==="
echo "- Use 'spike --help' to see all commands"
echo "- Use 'spike <command> --help' for command-specific help"
echo "- Configuration changes persist within a session"
echo "- Use presets for quick setup (development, experiment, production)"
echo "- Chat mode is great for exploratory conversations"