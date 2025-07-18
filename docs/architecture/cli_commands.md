# InsightSpike CLI Commands

## 🚀 Overview

InsightSpike provides two CLI interfaces:
- **`spike`** - Modern, user-friendly CLI (recommended)
- **`insightspike`** - Legacy CLI for backward compatibility

## 📋 Command Reference

### Core Commands

#### 1. **query** (`spike query` / `spike q`)
Ask questions and get insights from the knowledge base.

```bash
# Basic usage
spike query "What is machine learning?"

# With options
spike query "How does Python work?" --verbose
spike query "Explain neural networks" --preset experiment
```

**Options:**
- `--preset`: Config preset (development/experiment/production)
- `--verbose/-v`: Show detailed output including quality scores

**Aliases:** `q`, `ask`

---

#### 2. **embed** (`spike embed` / `spike e`)
Add documents to the knowledge base for learning.

```bash
# Add single file
spike embed document.txt

# Add directory of files
spike embed ./knowledge/

# With specific preset
spike embed data.md --preset production
```

**Options:**
- `--preset`: Config preset to use

**Aliases:** `e`, `learn`, `l`

---

#### 3. **interactive** (`spike interactive` / `spike chat`)
Enter interactive chat mode for continuous conversation.

```bash
spike interactive
# or
spike chat
```

**Interactive Commands:**
- Type any question to query
- `stats` - Show current statistics
- `clear` - Clear conversation history
- `help` - Show available commands
- `exit/quit/q` - Exit interactive mode

---

### Analysis Commands

#### 4. **insights** (`spike insights`)
Display discovered insights and statistics.

```bash
# Show recent insights
spike insights

# Show more insights
spike insights --limit 10
```

**Options:**
- `--limit`: Number of insights to display (default: 5)

---

#### 5. **insights-search** (`spike insights-search`)
Search for insights related to specific concepts.

```bash
spike insights-search "neural networks"
spike insights-search "quantum" --limit 20
```

**Options:**
- `--limit`: Maximum results to show (default: 10)

---

#### 6. **stats** (`spike stats`)
Show agent statistics and performance metrics.

```bash
spike stats
```

**Displays:**
- Initialization status
- Total reasoning cycles
- Average quality scores
- Memory statistics

---

### Experiment & Demo Commands

#### 7. **experiment** (`spike experiment`)
Run predefined experiments to test capabilities.

```bash
# Simple Q&A experiment
spike experiment --name simple

# Insight detection experiment
spike experiment --name insight --episodes 15

# Math reasoning experiment
spike experiment --name math
```

**Options:**
- `--name`: Experiment type (simple/insight/math)
- `--episodes`: Number of episodes to run (default: 10)

---

#### 8. **demo** (`spike demo`)
Run an interactive demo showcasing InsightSpike's capabilities.

```bash
spike demo
```

Demonstrates:
- Knowledge storage
- Retrieval accuracy
- Insight detection
- Complex reasoning

---

### Configuration Commands

#### 9. **config** (`spike config`)
Manage configuration settings.

```bash
# Show current configuration
spike config show

# Set a configuration value
spike config set retrieval.top_k 20
spike config set llm.temperature 0.5

# Save configuration
spike config save my_config.json

# Load configuration
spike config load my_config.json

# Apply preset
spike config preset experiment
```

**Actions:**
- `show` - Display current configuration
- `set <key> <value>` - Set configuration value
- `save [path]` - Save to file
- `load <path>` - Load from file
- `preset <name>` - Apply preset (development/experiment/production)

---

### Utility Commands

#### 10. **version** (`spike version`)
Display version information.

```bash
spike version
```

---

## 🎯 Common Workflows

### 1. Basic Knowledge Q&A
```bash
# Add knowledge
spike embed ./documents/

# Ask questions
spike query "What did I just add?"
spike query "Explain the main concepts" --verbose
```

### 2. Insight Discovery
```bash
# Add diverse knowledge
spike embed research_papers/
spike embed lecture_notes/

# Discover insights
spike query "How are these concepts related?"
spike insights
spike insights-search "quantum computing"
```

### 3. Interactive Learning Session
```bash
# Start interactive mode
spike chat

# In chat:
> What is machine learning?
> How does it relate to neuroscience?
> stats
> clear
> exit
```

### 4. Experimentation
```bash
# Test basic functionality
spike experiment --name simple

# Test insight detection
spike experiment --name insight --episodes 20

# Check statistics
spike stats
```

## 🔧 Configuration Presets

### development (default)
- Fast processing
- Lower quality thresholds
- Local models only
- Verbose logging

### experiment
- Balanced settings
- Medium thresholds
- Better model selection
- Suitable for testing

### production
- High quality thresholds
- Optimized for accuracy
- Full model capabilities
- Minimal logging

## 💡 Tips

1. **Use aliases for speed**: `spike q "question"` instead of `spike query`
2. **Interactive mode** is great for exploration and testing
3. **Verbose flag** helps understand reasoning quality
4. **Experiments** are useful for benchmarking performance
5. **Save configurations** for reproducible setups

## 🆘 Troubleshooting

If you encounter issues:
1. Check initialization: `spike stats`
2. Verify configuration: `spike config show`
3. Try different preset: `spike query "test" --preset experiment`
4. Clear state and retry: Start fresh with new agent instance