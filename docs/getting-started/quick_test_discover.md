# Quick Test Guide for InsightSpike CLI

## Testing InsightSpike Commands

### Basic Usage Test

```bash
# Test basic functionality
poetry run spike ask "What is entropy?"

# Add knowledge
poetry run spike add "Entropy measures disorder in a system"

# Test with mock provider (fast)
poetry run spike ask "Explain entropy" --provider mock
```

### 1. Testing Document Processing

```bash
# Create sample documents
mkdir -p data/test_discover
echo "Entropy is a measure of disorder and information content." > data/test_discover/entropy.txt
echo "Quantum effects appear in biological systems like photosynthesis." > data/test_discover/quantum_bio.txt
echo "Consciousness may involve information integration in the brain." > data/test_discover/consciousness.txt

# Load them (this might take a moment)
poetry run spike embed data/test_discover/
```

### 2. Testing Spike Detection

```bash
# Add related knowledge to trigger spike detection
poetry run spike add "Information theory relates entropy to uncertainty"
poetry run spike add "Thermodynamic entropy increases in isolated systems"

# Ask a question that should trigger spike detection
poetry run spike ask "How does entropy connect physics and information?"
```

### 3. Expected Spike Detection Output

When a spike is detected, you'll see:
```
🎯 SPIKE DETECTED!
├─ ΔGED: 0.35 (< 0.5) ✓
├─ ΔIG: 0.45 (> 0.2) ✓
└─ Confidence: High

💡 Insight: Connection between thermodynamic and information entropy discovered
```

## Performance Tips

1. **Use Mock Provider**: For testing, use `--provider mock` to avoid API calls
2. **Disable Message Passing**: Keep `enable_message_passing: false` for better performance
3. **Batch Operations**: Process multiple documents at once with `spike embed`

## Configuration Options

```yaml
# Spike detection thresholds
graph:
  spike_ged_threshold: 0.5  # Lower = more sensitive
  spike_ig_threshold: 0.2   # Higher = more sensitive

# Performance settings
processing:
  max_cycles: 10           # Reasoning cycles
  convergence_threshold: 0.8
```

## Troubleshooting

### "No spike detected"
- Add more related knowledge before asking
- Adjust thresholds in config.yaml
- Use `--verbose` to see metric values

### Slow Performance
- Ensure message passing is disabled
- Use mock provider for testing
- Check if PyTorch operations are CPU-bound

### API Errors
- Set API keys: `export OPENAI_API_KEY=...`
- Use mock provider: `--provider mock`
- Check network connectivity