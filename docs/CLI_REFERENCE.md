# InsightSpike-AI CLI Command Reference

This document provides comprehensive documentation for all CLI commands available in InsightSpike-AI.

## Installation and Setup

```bash
# Install InsightSpike-AI
pip install insightspike-ai

# Or install in development mode
git clone https://github.com/user/InsightSpike-AI.git
cd InsightSpike-AI
poetry install

# Verify installation
insightspike --version
```

## Global Options

All commands support these global options:

- `--help` - Show help message and exit
- `--version` - Show version information
- `--config PATH` - Specify custom config file path
- `--verbose` - Enable verbose logging
- `--quiet` - Suppress output except errors

## Core Commands

### `insightspike ask`

Interactive query interface for asking questions and discovering insights.

```bash
insightspike ask "What are the key insights about quantum computing?"
```

**Options:**
- `--mode MODE` - Processing mode: `fast`, `thorough`, `research` (default: `fast`)
- `--output FORMAT` - Output format: `text`, `json`, `markdown` (default: `text`)
- `--save PATH` - Save results to file
- `--context PATH` - Load additional context from file

**Examples:**
```bash
# Fast query
insightspike ask "Explain machine learning" --mode fast

# Thorough analysis with JSON output
insightspike ask "Complex systems theory" --mode thorough --output json

# Save results to file
insightspike ask "AI ethics" --save results.md --output markdown
```

### `insightspike query`

Programmatic query interface for batch processing.

```bash
insightspike query --input queries.txt --output results.json
```

**Options:**
- `--input PATH` - Input file with queries (one per line)
- `--output PATH` - Output file for results
- `--batch-size N` - Process N queries at once (default: 10)
- `--format FORMAT` - Output format: `json`, `csv`, `jsonl`

**Input File Format:**
```text
What is quantum computing?
Explain neural networks
How does blockchain work?
```

### `insightspike embed`

Generate embeddings for documents or text.

```bash
insightspike embed --input documents/ --output embeddings/
```

**Options:**
- `--input PATH` - Input directory or file
- `--output PATH` - Output directory for embeddings
- `--model MODEL` - Embedding model to use
- `--chunk-size N` - Text chunk size for processing
- `--format FORMAT` - Output format: `numpy`, `json`, `parquet`

### `insightspike load-documents`

Load and index documents into the system.

```bash
insightspike load-documents --source documents/ --index my_index
```

**Options:**
- `--source PATH` - Source directory with documents
- `--index NAME` - Index name to create
- `--recursive` - Process subdirectories recursively
- `--file-types TYPES` - File types to process (comma-separated)
- `--chunk-strategy STRATEGY` - Chunking strategy: `sentence`, `paragraph`, `page`

**Supported File Types:**
- Text: `.txt`, `.md`, `.rst`
- Documents: `.pdf`, `.docx`, `.html`
- Code: `.py`, `.js`, `.java`, `.cpp`
- Data: `.csv`, `.json`, `.xml`

## Analysis Commands

### `insightspike insights`

Discover insights from loaded documents.

```bash
insightspike insights --index my_index --topic "artificial intelligence"
```

**Options:**
- `--index NAME` - Index to analyze
- `--topic TOPIC` - Focus topic for insight discovery
- `--depth LEVEL` - Analysis depth: `shallow`, `medium`, `deep`
- `--output PATH` - Save insights to file
- `--threshold FLOAT` - Insight significance threshold (0.0-1.0)

### `insightspike insights-search`

Search for specific insights in the knowledge base.

```bash
insightspike insights-search "machine learning breakthroughs"
```

**Options:**
- `--query QUERY` - Search query
- `--limit N` - Maximum number of results
- `--index NAME` - Specific index to search
- `--relevance FLOAT` - Minimum relevance score

### `insightspike insights-validate`

Validate quality of discovered insights.

```bash
insightspike insights-validate --input insights.json --output validation_report.json
```

**Options:**
- `--input PATH` - Input insights file
- `--output PATH` - Output validation report
- `--metrics METRICS` - Validation metrics to apply
- `--threshold FLOAT` - Quality threshold

### `insightspike insights-cleanup`

Clean up and deduplicate insights.

```bash
insightspike insights-cleanup --index my_index --similarity-threshold 0.8
```

**Options:**
- `--index NAME` - Index to clean up
- `--similarity-threshold FLOAT` - Similarity threshold for deduplication
- `--dry-run` - Show what would be cleaned without doing it
- `--backup` - Create backup before cleanup

## Experimental Commands

### `insightspike experiment`

Run predefined experiments for research and validation.

```bash
insightspike experiment --name paradox_resolution --mode standard
```

**Options:**
- `--name NAME` - Experiment name to run
- `--mode MODE` - Experiment mode: `quick`, `standard`, `comprehensive`
- `--output-dir PATH` - Directory for experiment results
- `--config PATH` - Experiment configuration file

**Available Experiments:**
- `paradox_resolution` - Test insight discovery on paradoxes
- `scaffolded_learning` - Validate learning progression
- `cross_domain` - Cross-domain knowledge transfer
- `baseline_comparison` - Compare against baseline systems
- `real_time_detection` - Real-time insight detection

### `insightspike experiment-suite`

Run complete experiment suites for comprehensive validation.

```bash
insightspike experiment-suite --suite research_validation
```

**Options:**
- `--suite NAME` - Suite name: `research_validation`, `performance_benchmark`
- `--parallel N` - Number of parallel experiments
- `--timeout SECONDS` - Timeout per experiment

### `insightspike insight-experiment`

Interactive insight discovery experiment.

```bash
insightspike insight-experiment --dataset cognitive_puzzles
```

**Options:**
- `--dataset NAME` - Dataset for experiments
- `--interactive` - Enable interactive mode
- `--visualize` - Generate visualization outputs

## Performance Commands

### `insightspike benchmark`

Run performance benchmarks.

```bash
insightspike benchmark --suite performance --output benchmark_results.json
```

**Options:**
- `--suite NAME` - Benchmark suite: `performance`, `scalability`, `accuracy`
- `--iterations N` - Number of benchmark iterations
- `--output PATH` - Results output file
- `--compare-baseline` - Compare against baseline implementation

### `insightspike stats`

Show system statistics and performance metrics.

```bash
insightspike stats --index my_index
```

**Options:**
- `--index NAME` - Show stats for specific index
- `--detailed` - Show detailed breakdown
- `--format FORMAT` - Output format: `table`, `json`, `yaml`

### `insightspike compare-experiments`

Compare results from different experiments.

```bash
insightspike compare-experiments --exp1 results1.json --exp2 results2.json
```

**Options:**
- `--exp1 PATH` - First experiment results
- `--exp2 PATH` - Second experiment results
- `--metrics METRICS` - Metrics to compare
- `--output PATH` - Comparison report output

## Configuration Commands

### `insightspike config-info`

Display current configuration information.

```bash
insightspike config-info
```

**Options:**
- `--show-defaults` - Show default values
- `--validate` - Validate current configuration
- `--export PATH` - Export configuration to file

### `insightspike deps`

Manage dependencies and environment.

```bash
insightspike deps --check
```

**Options:**
- `--check` - Check dependency status
- `--install` - Install missing dependencies
- `--update` - Update dependencies
- `--platform` - Show platform-specific recommendations

## Development Commands

### `insightspike demo`

Run interactive demonstrations.

```bash
insightspike demo --type educational
```

**Options:**
- `--type TYPE` - Demo type: `educational`, `research`, `quick`
- `--interactive` - Enable interactive mode
- `--save-session PATH` - Save demo session

### `insightspike test-safe`

Run safe test commands for validation.

```bash
insightspike test-safe --component core
```

**Options:**
- `--component COMP` - Component to test: `core`, `embeddings`, `graph`
- `--quick` - Run quick tests only
- `--report PATH` - Generate test report

## Configuration File

Create a configuration file at `~/.insightspike/config.yaml`:

```yaml
# InsightSpike-AI Configuration
core:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  max_tokens: 512
  temperature: 0.7

memory:
  short_term_capacity: 10
  working_memory_capacity: 20
  episodic_memory_capacity: 60
  pattern_cache_capacity: 15

processing:
  batch_size: 32
  max_workers: 4
  timeout_seconds: 300

thresholds:
  ged_threshold: 0.5
  ig_threshold: 1.5
  insight_confidence: 0.8

output:
  default_format: "text"
  save_results: true
  generate_visualizations: false
```

## Environment Variables

Control behavior with environment variables:

```bash
export INSIGHTSPIKE_CONFIG="/path/to/config.yaml"
export INSIGHTSPIKE_LOG_LEVEL="INFO"
export INSIGHTSPIKE_CACHE_DIR="/tmp/insightspike"
export INSIGHTSPIKE_MAX_MEMORY="8GB"
export INSIGHTSPIKE_GPU_ENABLED="true"
```

## Examples and Use Cases

### Research Workflow

```bash
# 1. Load research papers
insightspike load-documents --source papers/ --index research_papers

# 2. Discover insights
insightspike insights --index research_papers --topic "quantum computing" --depth deep

# 3. Validate insights
insightspike insights-validate --input insights.json --output validation.json

# 4. Run experiments
insightspike experiment --name cross_domain --mode comprehensive
```

### Educational Use

```bash
# Interactive learning session
insightspike demo --type educational --interactive

# Ask educational questions
insightspike ask "Explain photosynthesis step by step" --mode thorough

# Load curriculum
insightspike load-documents --source curriculum/ --index biology_course
```

### Production Analysis

```bash
# Batch processing
insightspike query --input user_queries.txt --output analysis_results.json

# Performance monitoring
insightspike benchmark --suite performance --iterations 100

# System health check
insightspike stats --detailed --format json
```

## Troubleshooting

### Common Issues

**Command not found:**
```bash
# Ensure InsightSpike is installed and in PATH
which insightspike
pip install insightspike-ai
```

**Memory errors:**
```bash
# Reduce batch size
insightspike query --batch-size 5

# Set memory limit
export INSIGHTSPIKE_MAX_MEMORY="4GB"
```

**GPU issues:**
```bash
# Check GPU availability
insightspike deps --check

# Disable GPU
export INSIGHTSPIKE_GPU_ENABLED="false"
```

### Debug Mode

Enable detailed logging:

```bash
export INSIGHTSPIKE_LOG_LEVEL="DEBUG"
insightspike --verbose ask "test query"
```

### Getting Help

For additional help:

```bash
# Command-specific help
insightspike ask --help

# Show all commands
insightspike --help

# Version information
insightspike --version
```

## API Integration

For programmatic use, see the Python API:

```python
from insightspike import InsightSpikeAPI

# Initialize client
client = InsightSpikeAPI()

# Ask questions
result = client.ask("What is machine learning?")

# Discover insights
insights = client.discover_insights("artificial intelligence")

# Load documents
client.load_documents("path/to/docs", index_name="my_index")
```

---

For more information, visit the [InsightSpike-AI Documentation](https://github.com/user/InsightSpike-AI/docs).
