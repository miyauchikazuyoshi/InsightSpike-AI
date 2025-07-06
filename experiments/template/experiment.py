#!/usr/bin/env python3
"""
Experiment Template
==================

This is a template for creating new experiments in InsightSpike-AI.
Copy this template to start a new experiment.

Usage:
    1. Copy this entire template/ directory to a new experiment directory
    2. Rename it following the convention: YYYY-MM-DD_experiment_XX_description
    3. Update the experiment details below
    4. Run your experiment
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Now import InsightSpike modules
from src.insightspike.core.agents.main_agent import MainAgent
from src.insightspike.core.config import get_config


class ExperimentBase:
    """Base class for InsightSpike experiments."""
    
    def __init__(self, experiment_name: str, description: str):
        self.name = experiment_name
        self.description = description
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        self.exp_dir = Path(__file__).parent
        self.code_dir = self.exp_dir / "code"
        self.data_dir = self.exp_dir / "data"
        self.results_dir = self.exp_dir / "results"
        self.logs_dir = self.results_dir / "logs"
        
        # Create directories
        for dir_path in [self.code_dir, self.data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self._setup_logging()
        
        # Log experiment start
        logging.info(f"Starting experiment: {self.name}")
        logging.info(f"Description: {self.description}")
        logging.info(f"Timestamp: {self.timestamp}")
        
    def _setup_logging(self):
        """Setup logging for the experiment."""
        log_file = self.logs_dir / f"experiment_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def backup_main_data(self) -> Path:
        """Backup the main data directory."""
        main_data = PROJECT_ROOT / "data"
        backup_path = self.data_dir / f"backup_{self.timestamp}"
        
        logging.info(f"Backing up main data directory...")
        shutil.copytree(main_data, backup_path, dirs_exist_ok=True)
        logging.info(f"Backup saved to: {backup_path}")
        
        return backup_path
        
    def restore_main_data(self, backup_path: Path):
        """Restore the main data directory from backup."""
        main_data = PROJECT_ROOT / "data"
        
        logging.info(f"Restoring main data from: {backup_path}")
        if main_data.exists():
            shutil.rmtree(main_data)
        shutil.copytree(backup_path, main_data)
        logging.info("Main data restored successfully")
        
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save experiment results."""
        if filename is None:
            filename = f"results_{self.timestamp}.json"
            
        results_file = self.results_dir / filename
        
        # Add metadata
        results_with_metadata = {
            "experiment": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "results": results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
            
        logging.info(f"Results saved to: {results_file}")
        return results_file
        
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        config_file = self.exp_dir / "config.yaml"
        
        # For simplicity, save as JSON (you can use yaml if preferred)
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info(f"Config saved to: {config_file}")
        
    def run(self):
        """Main experiment logic - override this method."""
        raise NotImplementedError("Subclasses must implement the run() method")


# Example experiment implementation
class MyExperiment(ExperimentBase):
    """
    Example Experiment: [Your experiment name]
    
    Objective:
        [Describe what this experiment aims to achieve]
        
    Hypothesis:
        [What do you expect to discover/prove]
        
    Method:
        [Brief description of the approach]
    """
    
    def __init__(self):
        super().__init__(
            experiment_name="template_example",
            description="Template example showing how to structure experiments"
        )
        
    def run(self):
        """Run the experiment."""
        try:
            # 1. Setup
            logging.info("Setting up experiment...")
            config = get_config()
            
            # Optionally backup main data
            # backup_path = self.backup_main_data()
            
            # 2. Initialize system
            logging.info("Initializing InsightSpike system...")
            agent = MainAgent(config)
            agent.initialize()
            
            # 3. Run experiment
            logging.info("Running experiment...")
            results = {
                "episodes_processed": 0,
                "graph_nodes": 0,
                "insights_found": 0,
                "metrics": {}
            }
            
            # Your experiment code here
            # ...
            
            # 4. Save results
            self.save_results(results)
            
            # 5. Cleanup (if needed)
            # self.restore_main_data(backup_path)
            
            logging.info("Experiment completed successfully!")
            
        except Exception as e:
            logging.error(f"Experiment failed: {e}", exc_info=True)
            raise


def main():
    """Entry point for the experiment."""
    experiment = MyExperiment()
    experiment.run()


if __name__ == "__main__":
    main()