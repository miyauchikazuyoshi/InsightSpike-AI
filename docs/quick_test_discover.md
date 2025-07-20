# Quick Test Guide for spike discover

## Testing the discover command

Due to initialization timing, the discover command works best with pre-loaded knowledge. Here's how to test it:

### 1. First, load some sample documents

```bash
# Create sample documents
mkdir -p data/test_discover
echo "Entropy is a measure of disorder and information content." > data/test_discover/entropy.txt
echo "Quantum effects appear in biological systems like photosynthesis." > data/test_discover/quantum_bio.txt
echo "Consciousness may involve information integration in the brain." > data/test_discover/consciousness.txt

# Load them (this might take a moment)
poetry run spike embed data/test_discover/
```

### 2. Run discover on existing knowledge base

```bash
# Basic discovery
poetry run spike discover

# With options
poetry run spike discover --min-spike 0.5 --verbose

# Export results
poetry run spike discover --export insights.json
```

### 3. Expected Output

```
🔍 Discovering insights...

⚡ Discovered 3 insights

💡 Insight #1 [Spike: 0.72]
┌─────────────────────────────────────────────────────┐
│ Question: What is the relationship between entropy   │
│           and information?                          │
│ Confidence: 72%                                     │
└─────────────────────────────────────────────────────┘

🌉 Bridge Concepts:
┌─────────────┬───────────┬──────────────┐
│ Concept     │ Frequency │ Bridge Score │
├─────────────┼───────────┼──────────────┤
│ information │ 3         │ 0.60         │
│ quantum     │ 2         │ 0.40         │
└─────────────┴───────────┴──────────────┘
```

## Known Limitations

1. **Initialization delay**: The first command might take 10-20 seconds to initialize all components
2. **PyTorch dependency**: Full graph reasoning requires PyTorch (optional)
3. **Insight quality**: Better results with more documents in knowledge base

## Tips

- Start with a lower spike threshold (0.5) for initial testing
- The more documents you add, the better the insights
- Use `--verbose` to see detailed processing information