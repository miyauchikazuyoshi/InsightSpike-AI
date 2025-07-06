# InsightSpike-AI Experiments

This directory contains all experiments related to the InsightSpike-AI project, including active experiments, archived results, and supporting documentation.

## Quick Navigation

- [Active Experiments](#active-experiments)
- [Experiment Registry](#experiment-registry)
- [Data Handling Guide](./DATA_HANDLING_GUIDE.md)
- [Experiment Structure](./EXPERIMENT_STRUCTURE.md)
- [Directory Standards](./DIRECTORY_STRUCTURE_STANDARD.md)

## Directory Structure

```
experiments/
├── active_experiments/       # Currently running or recent experiments
├── archive/                  # Completed and compressed experiments
├── colab_experiments/        # Google Colab-specific experiments
├── data_preservation/        # Critical data backups
├── template/                 # Template for new experiments
└── [documentation files]     # Various guides and standards
```

## Active Experiments

### Current Experiments (July 2025)

| ID | Name | Status | Description | Start Date |
|----|------|--------|-------------|------------|
| 05 | [episode_management_graph_growth](./active_experiments/05_episode_management_graph_growth/) | Active | Episode management with graph growth analysis | 2025-07-04 |
| 06 | [huggingface_clean_dataset](./active_experiments/06_huggingface_clean_dataset/) | Completed | Clean dataset experiments with HuggingFace | 2025-07-04 |
| 07 | [squad_qa_alignment](./active_experiments/07_squad_qa_alignment/) | Completed | SQuAD QA alignment testing | 2025-07-04 |
| 08 | [large_scale_rag_1000](./active_experiments/08_large_scale_rag_1000/) | Active | Large-scale RAG with 1000 samples | 2025-07-05 |
| 09 | [rag_layer4_testing](./active_experiments/09_rag_layer4_testing/) | Completed | RAG Layer 4 functionality testing | 2025-07-05 |
| 10 | [data_backup_experiment](./active_experiments/10_data_backup_experiment/) | Active | Data backup and recovery procedures | 2025-07-06 |
| 11 | [graph_enhanced_rag_large](./active_experiments/11_graph_enhanced_rag_large/) | Active | Large-scale graph-enhanced RAG | 2025-07-06 |

## Key Documentation

### Essential Guides
- **[Data Handling Guide](./DATA_HANDLING_GUIDE.md)**: Comprehensive guide for experiment data management
- **[Experiment Structure](./EXPERIMENT_STRUCTURE.md)**: Standard structure for experiments
- **[Directory Standards](./DIRECTORY_STRUCTURE_STANDARD.md)**: Naming and organization standards

### Analysis and Findings
- **[Memory Usage Analysis](./MEMORY_USAGE_ANALYSIS.md)**: Memory optimization findings
- **[Compression Analysis](./INSIGHTSPIKE_COMPRESSION_ANALYSIS.md)**: Data compression strategies
- **[GED vs IG Comparison](./GED_IG_DEFINITIONS_COMPARISON.md)**: Graph Edit Distance vs Information Gain analysis

### Technical Specifications
- **[Experiment Registry](./EXPERIMENT_REGISTRY.md)**: Complete registry of all experiments
- **[Complex Maze Findings](./COMPLEX_MAZE_INITIAL_FINDINGS.md)**: Initial maze navigation results
- **[Objective Function Clarification](./OBJECTIVE_FUNCTION_CLARIFICATION.md)**: Core objective functions

## Quick Start for New Experiments

### 1. Create New Experiment

```bash
# Copy template
cp -r experiments/template experiments/active_experiments/XX_your_experiment_name

# Navigate to your experiment
cd experiments/active_experiments/XX_your_experiment_name

# Update README with your experiment details
```

### 2. Follow Standards

1. **Naming**: Use format `XX_descriptive_name` (e.g., `12_attention_mechanism_test`)
2. **Structure**: Follow the [standard directory structure](./EXPERIMENT_STRUCTURE.md)
3. **Documentation**: Create README.md with objectives, methods, and results
4. **Data**: Follow the [data handling guide](./DATA_HANDLING_GUIDE.md)

### 3. Run Experiment

```python
# Example experiment script structure
from insightspike.core import GraphMemory
from insightspike.graph_algorithms import GraphGrowthExperiment

# Load your configuration
config = load_config("config.yaml")

# Initialize experiment
experiment = GraphGrowthExperiment(config)

# Run with proper logging
results = experiment.run(log_dir="logs/")

# Save results
save_results(results, "results/")
```

## Best Practices

### Version Control
- **Do**: Commit code, configs, small results, documentation
- **Don't**: Commit large data files, checkpoints, temporary files
- **Use**: Git LFS for essential large files

### Data Management
1. Always backup data before experiments
2. Use checksums for data integrity
3. Document data sources and preprocessing
4. Clean up after experiments

### Documentation
- Update experiment README throughout the process
- Include visualizations in results/
- Document failures and lessons learned
- Link related experiments

## Utilities and Scripts

### Common Tasks

```bash
# Validate experiment structure
python scripts/validate_experiment.py XX_experiment_name

# Backup experiment data
./scripts/backup_experiment.sh XX_experiment_name

# Archive completed experiment
./scripts/archive_experiment.sh XX_experiment_name

# Generate experiment report
python scripts/generate_report.py XX_experiment_name
```

### Analysis Tools

```python
# Compare experiments
from experiments.utils import compare_experiments
compare_experiments(['05', '08', '11'], metrics=['memory', 'speed', 'accuracy'])

# Visualize results
from experiments.utils import plot_results
plot_results('active_experiments/08_large_scale_rag_1000/results/')
```

## Experiment Lifecycle

```mermaid
graph LR
    A[Planning] --> B[Setup]
    B --> C[Active Development]
    C --> D[Execution]
    D --> E[Analysis]
    E --> F[Documentation]
    F --> G[Review]
    G --> H[Archive]
```

### Phases:
1. **Planning** (1-2 days): Define objectives, design experiment
2. **Setup** (1-2 days): Prepare data, configure environment
3. **Active Development** (1-2 weeks): Implement and iterate
4. **Execution** (varies): Run full experiment
5. **Analysis** (2-3 days): Analyze results, create visualizations
6. **Documentation** (1-2 days): Write up findings
7. **Review** (1 week): Peer review, validation
8. **Archive** (1 day): Compress and move to archive/

## Collaboration

### Working on Experiments
1. Create a branch for major experiments
2. Use descriptive commit messages
3. Update experiment README regularly
4. Share findings in team meetings

### Code Reviews
- Focus on reproducibility
- Check data handling practices
- Validate statistical methods
- Ensure proper documentation

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Use data generators, reduce batch size |
| Slow execution | Profile code, use caching, parallelize |
| Data corruption | Check checksums, restore from backup |
| Can't reproduce | Verify seeds, environment, data version |

### Getting Help
1. Check existing experiment logs
2. Review similar experiments
3. Consult documentation
4. Ask in team chat/meetings

## Future Experiments

### Planned
- Attention mechanism integration
- Multi-modal memory systems
- Distributed graph processing
- Real-time learning capabilities

### Proposed
- Cross-domain transfer learning
- Adversarial robustness testing
- Explainability analysis
- Efficiency optimizations

## Contributing

1. **Propose**: Create issue with experiment proposal
2. **Discuss**: Get feedback from team
3. **Implement**: Follow standards and guidelines
4. **Document**: Thorough documentation throughout
5. **Share**: Present findings to team

## Resources

### Internal
- [Main Project README](../README.md)
- [API Documentation](../docs/api/)
- [Architecture Overview](../docs/architecture.md)

### External
- [Graph Neural Networks Paper](https://arxiv.org/...)
- [RAG Systems Survey](https://arxiv.org/...)
- [Memory Systems in AI](https://arxiv.org/...)

---

For questions or suggestions, please contact the InsightSpike-AI team or create an issue in the repository.