#!/usr/bin/env python3
"""
Config Migration Guide Generator
================================

Generates a comprehensive guide for migrating from legacy_config to InsightSpikeConfig.
Includes mapping tables, code examples, and common patterns.
"""

from pathlib import Path
from typing import Dict, List, Tuple


class MigrationGuideGenerator:
    """Generates migration documentation"""
    
    def __init__(self):
        self.mapping = self._create_attribute_mapping()
        self.examples = self._create_code_examples()
        
    def _create_attribute_mapping(self) -> Dict[str, str]:
        """Create comprehensive attribute mapping"""
        return {
            # Core/Embedding/LLM mappings
            'config.core.model_name': 'config.core.model_name',
            'config.core.device': 'config.core.device',
            'config.core.provider': 'config.core.llm_provider',
            'config.core.model_name': 'config.core.llm_model',
            'config.core.max_tokens': 'config.core.max_tokens',
            'config.core.temperature': 'config.core.temperature',
            'config.core.device': 'config.core.device',
            'config.core.use_gpu': 'config.core.use_gpu',
            'config.core.safe_mode': 'config.core.safe_mode',
            
            # Memory mappings
            'config.memory.max_retrieved_docs': 'config.memory.max_retrieved_docs',
            'config.memory.merge_ged': 'config.reasoning.episode_merge_threshold',
            'config.memory.split_ig': 'config.reasoning.episode_split_threshold',
            'config.memory.prune_c': 'config.reasoning.episode_prune_threshold',
            
            # Retrieval mappings
            'config.retrieval.similarity_threshold': 'config.retrieval.similarity_threshold',
            'config.retrieval.top_k': 'config.retrieval.top_k',
            'config.retrieval.layer1_top_k': 'config.retrieval.layer1_top_k',
            'config.retrieval.layer2_top_k': 'config.retrieval.layer2_top_k',
            'config.retrieval.layer3_top_k': 'config.retrieval.layer3_top_k',
            
            # Graph mappings
            'config.graph.spike_ged_threshold': 'config.graph.spike_ged_threshold',
            'config.graph.spike_ig_threshold': 'config.graph.spike_ig_threshold',
            'config.graph.ged_algorithm': 'config.graph.ged_algorithm',
            'config.graph.ig_algorithm': 'config.graph.ig_algorithm',
            
            # Reasoning mappings
            'config.reasoning.use_gnn': 'config.graph.use_gnn',
            'config.reasoning.gnn_hidden_dim': 'config.graph.gnn_hidden_dim',
            'config.reasoning.weight_ged': 'config.reasoning.weight_ged',
            'config.reasoning.weight_ig': 'config.reasoning.weight_ig',
            'config.reasoning.weight_conflict': 'config.reasoning.weight_conflict',
            'config.reasoning.conflict_threshold': 'config.reasoning.conflict_threshold',
            
            # Spike mappings
            'config.spike.spike_ged': 'config.spike.spike_ged',
            'config.spike.spike_ig': 'config.spike.spike_ig',
            'config.spike.eta_spike': 'config.spike.eta_spike',
            
            # Path mappings
            'config.paths.data_dir': 'config.paths.data_dir',
            'config.paths.log_dir': 'config.paths.log_dir',
            'config.paths.index_file': 'config.paths.index_file',
            'config.paths.graph_file': 'config.paths.graph_file',
        }
    
    def _create_code_examples(self) -> List[Tuple[str, str, str]]:
        """Create before/after code examples"""
        return [
            (
                "Basic Import",
                """from insightspike.config import InsightSpikeConfig

config = InsightSpikeConfig()""",
                """from insightspike.config import InsightSpikeConfig

config = InsightSpikeConfig()"""
            ),
            (
                "Import with get_config",
                """from insightspike.config import get_config

config = get_config()""",
                """from insightspike.config import get_config

config = get_config()"""
            ),
            (
                "Setting LLM Parameters",
                """config = InsightSpikeConfig()
config.core.model_name = "gpt-3.5-turbo"
config.core.provider = "openai"
config.core.max_tokens = 512""",
                """config = InsightSpikeConfig()
config.core.llm_model = "gpt-3.5-turbo"
config.core.llm_provider = "openai"
config.core.max_tokens = 512"""
            ),
            (
                "Accessing Embedding Config",
                """model_name = config.core.model_name
device = config.core.device""",
                """model_name = config.core.model_name
device = config.core.device"""
            ),
            (
                "Creating Config with Parameters",
                """from insightspike.config.legacy_config import (
    Config, EmbeddingConfig, LLMConfig
)

config = Config(
    embedding=EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    llm=LLMConfig(
        provider="local",
        model_name="distilgpt2"
    )
)""",
                """from insightspike.config import InsightSpikeConfig

config = InsightSpikeConfig(
    core={
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_provider": "local",
        "llm_model": "distilgpt2"
    }
)"""
            ),
            (
                "Loading from YAML",
                """# Legacy system didn't have built-in YAML support
config = InsightSpikeConfig()
# Manual configuration needed""",
                """from insightspike.config import load_config

# Load from default config.yaml
config = load_config()

# Or load from specific file
config = load_config("path/to/config.yaml")"""
            ),
            (
                "Using Safe Mode",
                """config = InsightSpikeConfig()
config.core.safe_mode = True""",
                """config = InsightSpikeConfig()
config.core.safe_mode = True"""
            ),
            (
                "Accessing Graph Config",
                """ged_threshold = config.graph.spike_ged_threshold
use_gnn = config.reasoning.use_gnn""",
                """ged_threshold = config.graph.spike_ged_threshold
use_gnn = config.graph.use_gnn  # Note: moved from reasoning to graph"""
            )
        ]
    
    def generate_markdown_guide(self) -> str:
        """Generate a markdown migration guide"""
        lines = [
            "# Config Migration Guide: legacy_config to InsightSpikeConfig",
            "",
            "## Overview",
            "",
            "This guide helps you migrate from the old `legacy_config` system to the new `InsightSpikeConfig` system.",
            "",
            "## Key Changes",
            "",
            "1. **Import Changes**: `Config` is now `InsightSpikeConfig`",
            "2. **Structure Changes**: Some attributes have moved between sections",
            "3. **New Features**: YAML/JSON support, environment variables, validation",
            "4. **Naming Changes**: Some attributes have been renamed for clarity",
            "",
            "## Import Migration",
            "",
            "### Old Imports",
            "```python",
            "from insightspike.config import InsightSpikeConfig",
            "from insightspike.config import get_config",
            "from insightspike.config.legacy_config import (",
            "    EmbeddingConfig, LLMConfig, RetrievalConfig",
            ")",
            "```",
            "",
            "### New Imports",
            "```python",
            "from insightspike.config import InsightSpikeConfig",
            "from insightspike.config import get_config, load_config",
            "from insightspike.config.models import (",
            "    CoreConfig, MemoryConfig, RetrievalConfig",
            ")",
            "```",
            "",
            "## Attribute Mapping Table",
            "",
            "| Old Path | New Path | Notes |",
            "|----------|----------|-------|"
        ]
        
        # Add mapping table
        for old_path, new_path in sorted(self.mapping.items()):
            notes = ""
            if "llm.model_name" in old_path:
                notes = "Renamed for clarity"
            elif "reasoning.use_gnn" in old_path:
                notes = "Moved to graph section"
            elif "memory.merge_ged" in old_path:
                notes = "Moved to reasoning.episode_*"
            
            lines.append(f"| `{old_path}` | `{new_path}` | {notes} |")
        
        lines.extend([
            "",
            "## Code Examples",
            ""
        ])
        
        # Add code examples
        for title, before, after in self.examples:
            lines.extend([
                f"### {title}",
                "",
                "**Before (legacy_config):**",
                "```python",
                before.strip(),
                "```",
                "",
                "**After (InsightSpikeConfig):**",
                "```python",
                after.strip(),
                "```",
                ""
            ])
        
        # Add common patterns section
        lines.extend([
            "## Common Migration Patterns",
            "",
            "### 1. Config Instantiation",
            "",
            "- Replace `Config()` with `InsightSpikeConfig()`",
            "- Use `load_config()` to load from YAML files",
            "- Use `get_config()` for default configuration",
            "",
            "### 2. Nested Config Objects",
            "",
            "The new system uses flat dictionaries instead of nested dataclasses:",
            "",
            "```python",
            "# Old way",
            "config = Config(",
            "    llm=LLMConfig(provider='openai', model_name='gpt-4')",
            ")",
            "",
            "# New way",
            "config = InsightSpikeConfig(",
            "    core={'llm_provider': 'openai', 'llm_model': 'gpt-4'}",
            ")",
            "```",
            "",
            "### 3. Environment Variables",
            "",
            "The new system supports environment variable overrides:",
            "",
            "```bash",
            "export INSIGHTSPIKE_CORE_LLM_PROVIDER=openai",
            "export INSIGHTSPIKE_CORE_LLM_MODEL=gpt-4",
            "```",
            "",
            "### 4. Validation",
            "",
            "The new system uses Pydantic for validation:",
            "",
            "```python",
            "try:",
            "    config = InsightSpikeConfig(",
            "        core={'temperature': 3.0}  # Will fail validation (max is 2.0)",
            "    )",
            "except ValidationError as e:",
            "    print(f\"Invalid config: {e}\")",
            "```",
            "",
            "## Deprecated Features",
            "",
            "The following features are deprecated or removed:",
            "",
            "1. `config.core.dimension` - Now inferred from model",
            "2. `config.memory.inactive_n` - No longer used",
            "3. `config.paths.root_dir` - Handled differently",
            "4. Direct dataclass instantiation - Use dictionaries instead",
            "",
            "## Migration Checklist",
            "",
            "- [ ] Update all imports from `legacy_config` to new config system",
            "- [ ] Replace `Config()` with `InsightSpikeConfig()`",
            "- [ ] Update attribute access paths according to mapping table",
            "- [ ] Replace nested dataclass configs with dictionaries",
            "- [ ] Test with `--dry-run` option first",
            "- [ ] Run full test suite after migration",
            "- [ ] Update any custom config extensions",
            "- [ ] Update documentation and comments",
            "",
            "## Troubleshooting",
            "",
            "### Import Errors",
            "",
            "If you get import errors, ensure you're using:",
            "```python",
            "from insightspike.config import InsightSpikeConfig",
            "```",
            "NOT:",
            "```python",
            "from insightspike.config.models import InsightSpikeConfig  # Wrong!",
            "```",
            "",
            "### Attribute Errors",
            "",
            "If you get attribute errors like `'InsightSpikeConfig' has no attribute 'llm'`, ",
            "remember that the structure has changed. Use `config.core.llm_model` instead ",
            "of `config.core.model_name`.",
            "",
            "### Validation Errors",
            "",
            "The new system has stricter validation. Common issues:",
            "- Temperature must be between 0.0 and 2.0",
            "- Similarity thresholds must be between 0.0 and 1.0",
            "- File paths must be valid Path objects",
            "",
            "## Getting Help",
            "",
            "1. Run the migration script with `--dry-run` to preview changes",
            "2. Check the generated report for warnings and errors",
            "3. Use the attribute mapping table above for manual updates",
            "4. Refer to the new config schema in `config/models.py`"
        ])
        
        return "\n".join(lines)
    
    def generate_html_guide(self) -> str:
        """Generate an HTML version of the guide"""
        markdown_content = self.generate_markdown_guide()
        
        # Simple HTML wrapper (in production, use a proper markdown parser)
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Config Migration Guide</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{ color: #333; }}
        code {{ 
            background: #f4f4f4; 
            padding: 2px 4px; 
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{ 
            background: #f4f4f4; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left;
        }}
        th {{ 
            background: #f8f8f8; 
            font-weight: bold;
        }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .deprecated {{ 
            background: #fff3cd; 
            padding: 10px; 
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }}
        .note {{ 
            background: #d4edda; 
            padding: 10px; 
            border-left: 4px solid #28a745;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="content">
        <!-- Markdown content would be parsed here -->
        <h1>Config Migration Guide</h1>
        <p>This is a placeholder. Use a proper markdown parser for production.</p>
    </div>
</body>
</html>"""
        
        return html


def main():
    """Generate migration guides"""
    generator = MigrationGuideGenerator()
    
    # Generate markdown guide
    markdown_guide = generator.generate_markdown_guide()
    
    # Save to file
    output_path = Path("CONFIG_MIGRATION_GUIDE.md")
    output_path.write_text(markdown_guide)
    
    print(f"✅ Migration guide generated: {output_path}")
    print(f"📄 Size: {len(markdown_guide)} characters")
    print(f"📊 Mappings documented: {len(generator.mapping)}")
    print(f"💡 Examples provided: {len(generator.examples)}")
    print("\nYou can now:")
    print("  1. Review the guide at CONFIG_MIGRATION_GUIDE.md")
    print("  2. Run the migration script: python scripts/migrate_config.py")
    print("  3. Use --dry-run option to preview changes first")


if __name__ == "__main__":
    main()