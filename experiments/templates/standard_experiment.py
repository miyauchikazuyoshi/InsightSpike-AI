#!/usr/bin/env python3
"""
Standard Experiment Template for InsightSpike
============================================

This template ensures consistent experiment design and execution.

IMPORTANT: Follow guidelines in /CLAUDE.md when implementing experiments.
Specifically:
- Do not include answers directly in knowledge base
- Use genuine processing (no mocks)
- Maintain data integrity
- Follow the standard experiment structure
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import yaml

# InsightSpike imports
from insightspike.config import load_config, InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
from insightspike.config.converter import ConfigConverter
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.factory import create_datastore


class StandardExperiment(ABC):
    """Base class for all InsightSpike experiments"""
    
    def __init__(self, experiment_name: str, config_path: Optional[str] = None, preset: str = "experiment"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup paths
        self.base_path = Path(__file__).parent.parent / experiment_name
        self.data_path = self.base_path / "data"
        self.results_path = self.base_path / "results" / self.timestamp
        self.results_path.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load InsightSpike config using new system
        if config_path:
            self.insight_config = load_config(config_path=config_path)
        else:
            self.insight_config = load_config(preset=preset)
            
        # Create datastore and agent
        self.datastore = create_datastore()
        self.agent = self._create_agent()
        
        # Load experiment-specific config if provided
        self.experiment_config = self._load_experiment_config(config_path)
        
        # Validate setup
        self._validate_setup()
        
    def _setup_logging(self):
        """Setup standardized logging"""
        log_path = self.results_path / "experiment.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
        
    def _create_agent(self) -> MainAgent:
        """Create InsightSpike agent with current config"""
        # Convert Pydantic config to legacy format for MainAgent
        config_dict = self.insight_config.dict()
        legacy_config = ConfigConverter.preset_dict_to_legacy_config(config_dict)
        
        agent = MainAgent(config=legacy_config, datastore=self.datastore)
        if not agent.initialize():
            raise RuntimeError("Failed to initialize InsightSpike agent")
        return agent
        
    def _load_experiment_config(self, config_path: Optional[str]) -> Dict:
        """Load experiment-specific configuration (not InsightSpike config)"""
        if config_path and Path(config_path).exists() and config_path.endswith('.yaml'):
            # This is for experiment-specific settings, not InsightSpike config
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
        
    def _validate_setup(self):
        """Validate experiment setup"""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        
        # Check data integrity
        if not self._check_data_integrity():
            raise ValueError("Data integrity check failed!")
            
        self.logger.info("Setup validation passed ✓")
        
    @abstractmethod
    def _check_data_integrity(self) -> bool:
        """
        Check data integrity (no cheating, proper format, etc.)
        Must be implemented by each experiment.
        """
        pass
        
    @abstractmethod
    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare data for the experiment.
        Returns dict with 'train', 'test', and 'metadata'.
        """
        pass
        
    @abstractmethod
    def run_baseline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run baseline method(s).
        Returns results dictionary.
        """
        pass
        
    @abstractmethod
    def run_proposed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run proposed method (InsightSpike).
        Returns results dictionary.
        """
        pass
        
    @abstractmethod
    def evaluate_results(self, baseline: Dict, proposed: Dict) -> Dict[str, Any]:
        """
        Evaluate and compare results.
        Returns metrics dictionary.
        """
        pass
        
    def run(self):
        """Standard experiment execution flow"""
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT: {self.experiment_name}")
        self.logger.info("="*60)
        
        try:
            # 1. Prepare data
            self.logger.info("\n1. Preparing data...")
            data = self.prepare_data()
            self._save_data_info(data)
            
            # 2. Run baseline
            self.logger.info("\n2. Running baseline...")
            baseline_results = self.run_baseline(data)
            self._save_results(baseline_results, "baseline")
            
            # 3. Run proposed method
            self.logger.info("\n3. Running proposed method...")
            proposed_results = self.run_proposed(data)
            self._save_results(proposed_results, "proposed")
            
            # 4. Evaluate
            self.logger.info("\n4. Evaluating results...")
            metrics = self.evaluate_results(baseline_results, proposed_results)
            self._save_metrics(metrics)
            
            # 5. Generate report
            self.logger.info("\n5. Generating report...")
            report = self._generate_report(data, baseline_results, proposed_results, metrics)
            self._save_report(report)
            
            self.logger.info("\n✅ Experiment completed successfully!")
            self.logger.info(f"Results saved to: {self.results_path}")
            
            return {
                'success': True,
                'metrics': metrics,
                'report_path': str(self.results_path / "report.md")
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
            
    def _save_data_info(self, data: Dict):
        """Save data information for reproducibility"""
        info = {
            'timestamp': self.timestamp,
            'data_hash': self._hash_data(data),
            'metadata': data.get('metadata', {}),
            'statistics': {
                'train_size': len(data.get('train', [])),
                'test_size': len(data.get('test', []))
            }
        }
        
        with open(self.results_path / "data_info.json", 'w') as f:
            json.dump(info, f, indent=2)
            
    def _hash_data(self, data: Any) -> str:
        """Create hash of data for integrity checking"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
        
    def _save_results(self, results: Dict, name: str):
        """Save results to file"""
        with open(self.results_path / f"{name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
    def _save_metrics(self, metrics: Dict):
        """Save metrics to file"""
        with open(self.results_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def _generate_report(self, data: Dict, baseline: Dict, proposed: Dict, metrics: Dict) -> str:
        """Generate markdown report"""
        report = f"""# Experiment Report: {self.experiment_name}

## Experiment Information
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Timestamp**: {self.timestamp}
- **InsightSpike Config**: {self.insight_config.dict()}
- **Experiment Config**: {json.dumps(self.experiment_config, indent=2)}

## Data Summary
- **Test Size**: {len(data.get('test', []))}
- **Data Hash**: {self._hash_data(data)[:16]}...

## Results

### Baseline Performance
{self._format_results(baseline)}

### Proposed Method Performance
{self._format_results(proposed)}

## Metrics Comparison
{self._format_metrics(metrics)}

## Conclusion
{metrics.get('summary', 'No summary provided')}
"""
        return report
        
    def _format_results(self, results: Dict) -> str:
        """Format results for report"""
        # Override in subclass for custom formatting
        return f"```json\n{json.dumps(results, indent=2)}\n```"
        
    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for report"""
        # Override in subclass for custom formatting
        lines = []
        for key, value in metrics.items():
            if key != 'summary':
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)
        
    def _save_report(self, report: str):
        """Save report to file"""
        with open(self.results_path / "report.md", 'w') as f:
            f.write(report)