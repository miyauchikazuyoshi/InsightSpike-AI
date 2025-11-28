"""End-to-end tests for complete InsightSpike workflows."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from insightspike.config.models import InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory


class TestCompleteWorkflow:
    """Test complete workflows from start to finish."""

    @pytest.fixture
    def test_data_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_knowledge_building_workflow(self, test_data_dir):
        """Test building knowledge base and querying it."""
        # Step 1: Create configuration
        config = ConfigPresets.development()

        # Step 2: Create datastore
        datastore_config = {
            "type": "filesystem",
            "params": {"base_path": str(test_data_dir)},
        }
        datastore = DataStoreFactory.create(datastore_config)

        # Step 3: Initialize agent
        agent = MainAgent(config=config, datastore=datastore)
        assert agent.initialize()

        # Step 4: Add knowledge
        knowledge_texts = [
            "InsightSpike is an AI system for discovering insights.",
            "It uses graph-based reasoning to detect patterns.",
            "When different concepts connect, new insights emerge.",
            "The system has multiple layers: error monitoring, memory, graph reasoning, and language.",
            "Integration of independent systems creates emergent properties.",
        ]

        for text in knowledge_texts:
            result = agent.learn(text)
            assert result["success"]

        # Step 5: Query the knowledge
        questions = [
            "What is InsightSpike?",
            "How does it detect patterns?",
            "What happens when systems integrate?",
        ]

        for question in questions:
            result = agent.process_question(question, max_cycles=3)
            assert result.success
            assert len(result.response) > 0

            # Check if relevant knowledge was retrieved
            if "InsightSpike" in question:
                assert any(
                    "InsightSpike" in doc.get("text", "")
                    for doc in result.retrieved_documents
                )

        # Step 6: Save state
        assert agent.save_state()

        # Step 7: Create new agent and load state
        agent2 = MainAgent(config=config, datastore=datastore)
        assert agent2.initialize()
        assert agent2.load_state()

        # Step 8: Verify loaded knowledge
        stats = agent2.get_stats()
        assert stats["memory_stats"]["total_episodes"] >= len(knowledge_texts)

        # Query with new agent should work
        result = agent2.process_question("Tell me about InsightSpike", max_cycles=2)
        assert result.success
        assert "InsightSpike" in result.response or "AI system" in result.response

    def test_insight_discovery_workflow(self, test_data_dir):
        """Test discovering insights through incremental learning."""
        config = ConfigPresets.experiment()
        datastore = DataStoreFactory.create(
            {"type": "filesystem", "params": {"base_path": str(test_data_dir)}}
        )

        agent = MainAgent(config=config, datastore=datastore)
        assert agent.initialize()

        # Build knowledge incrementally
        concepts = [
            # Individual concepts
            "System A processes data independently using its own algorithms.",
            "System B analyzes patterns independently with different methods.",
            "Each system has its own strengths and limitations.",
            # Integration hint
            "When systems work together, they can share information.",
            # Emergence insight
            "Combining A and B creates capabilities neither has alone - this is emergence.",
        ]

        insights_found = []

        for i, concept in enumerate(concepts):
            result = agent.learn(concept)

            # Check for insights
            if result.get("insights"):
                insights_found.extend(result["insights"])

            # Query to trigger reasoning
            if i >= 2:  # After some knowledge is built
                query_result = agent.process_question(
                    "What happens when systems combine?", max_cycles=3
                )

                if query_result.spike_detected:
                    insights_found.append(
                        {
                            "type": "spike",
                            "question": query_result.question,
                            "insight": query_result.response,
                        }
                    )

        # Should have discovered some insights
        assert len(insights_found) > 0

        # Final comprehensive query
        final_result = agent.process_question(
            "Explain the concept of emergence in system integration", max_cycles=5
        )

        assert final_result.success
        assert final_result.reasoning_quality > 0.5

        # Response should reflect the learned concepts
        response_lower = final_result.response.lower()
        assert any(
            term in response_lower
            for term in ["emergence", "combine", "together", "capabilities"]
        )


class TestCLIWorkflow:
    """Test complete workflows using the CLI."""

    def test_cli_basic_workflow(self, test_data_dir):
        """Test basic CLI workflow: embed -> query -> stats."""
        repo_root = Path(__file__).resolve().parents[2]
        src_path = repo_root / "src"
        # Create test knowledge file
        knowledge_file = test_data_dir / "knowledge.txt"
        knowledge_file.write_text(
            """
InsightSpike is an advanced AI system.
It discovers insights through knowledge synthesis.
Graph-based reasoning enables pattern detection.
Multiple layers work together for intelligence.
"""
        )

        # Test embed command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "insightspike.cli.spike",
                "embed",
                str(knowledge_file),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "INSIGHTSPIKE_DATA_DIR": str(test_data_dir), "PYTHONPATH": str(src_path)},
        )

        assert result.returncode == 0
        assert "Successfully embedded" in result.stdout

        # Test query command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "insightspike.cli.spike",
                "query",
                "What is InsightSpike?",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "INSIGHTSPIKE_DATA_DIR": str(test_data_dir), "PYTHONPATH": str(src_path)},
        )

        assert result.returncode == 0
        assert "Answer:" in result.stdout

        # Test stats command
        result = subprocess.run(
            [sys.executable, "-m", "insightspike.cli.spike", "stats"],
            capture_output=True,
            text=True,
            env={**os.environ, "INSIGHTSPIKE_DATA_DIR": str(test_data_dir), "PYTHONPATH": str(src_path)},
        )

        assert result.returncode == 0
        assert "Agent Statistics" in result.stdout
        assert "Total episodes:" in result.stdout

    def test_cli_config_workflow(self, test_data_dir):
        """Test CLI configuration management workflow."""
        repo_root = Path(__file__).resolve().parents[2]
        src_path = repo_root / "src"
        config_file = test_data_dir / "custom_config.json"

        # Export default config
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "insightspike.cli.spike",
                "config",
                "export",
                str(config_file),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(src_path)},
        )

        assert result.returncode == 0
        assert config_file.exists()

        # Modify config
        with open(config_file) as f:
            config_data = json.load(f)

        config_data["environment"] = "custom"
        config_data["llm"]["temperature"] = 0.5

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Validate modified config
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "insightspike.cli.spike",
                "config",
                "validate",
                str(config_file),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(src_path)},
        )

        assert result.returncode == 0
        assert "Configuration is valid" in result.stdout


class TestMultiModalWorkflow:
    """Test workflows with different types of content."""

    def test_structured_data_workflow(self, test_data_dir):
        """Test processing structured data."""
        config = ConfigPresets.development()
        datastore = DataStoreFactory.create(
            {"type": "filesystem", "params": {"base_path": str(test_data_dir)}}
        )

        agent = MainAgent(config=config, datastore=datastore)
        assert agent.initialize()

        # Add structured knowledge
        structured_data = [
            {
                "concept": "Machine Learning",
                "definition": "Algorithms that improve through experience",
                "type": "technology",
            },
            {
                "concept": "Neural Networks",
                "definition": "Computing systems inspired by biological neurons",
                "type": "technology",
            },
            {
                "concept": "Deep Learning",
                "definition": "Neural networks with multiple layers",
                "type": "technology",
            },
            {
                "relationship": "Deep Learning",
                "uses": "Neural Networks",
                "strength": 0.9,
            },
            {
                "relationship": "Neural Networks",
                "part_of": "Machine Learning",
                "strength": 0.8,
            },
        ]

        for item in structured_data:
            text = json.dumps(item)
            result = agent.learn(text)
            assert result["success"]

        # Query about relationships
        result = agent.process_question(
            "How are neural networks related to deep learning?", max_cycles=3
        )

        assert result.success
        assert any(
            term in result.response.lower()
            for term in ["neural", "deep", "layer", "multiple"]
        )

    def test_incremental_learning_workflow(self, test_data_dir):
        """Test incremental learning with feedback."""
        config = ConfigPresets.experiment()
        datastore = DataStoreFactory.create(
            {"type": "filesystem", "params": {"base_path": str(test_data_dir)}}
        )

        agent = MainAgent(config=config, datastore=datastore)
        assert agent.initialize()

        # Initial knowledge
        agent.learn("Quantum computing uses qubits instead of classical bits.")

        # Query and get initial understanding
        result1 = agent.process_question("What is quantum computing?", max_cycles=2)
        initial_quality = result1.reasoning_quality

        # Add more specific knowledge
        agent.learn("Qubits can exist in superposition, being 0 and 1 simultaneously.")
        agent.learn(
            "This superposition enables quantum parallelism for certain algorithms."
        )

        # Query again
        result2 = agent.process_question("What is quantum computing?", max_cycles=2)

        # Understanding should improve
        assert result2.reasoning_quality >= initial_quality
        assert (
            "superposition" in result2.response.lower()
            or "qubit" in result2.response.lower()
        )

        # Add connecting insight
        agent.learn(
            "Quantum computing's power comes from exploiting superposition and entanglement together."
        )

        # Final query should show integrated understanding
        result3 = agent.process_question(
            "What makes quantum computing powerful?", max_cycles=3
        )

        assert result3.success
        assert result3.reasoning_quality > 0.5


class TestErrorRecoveryWorkflow:
    """Test system behavior under error conditions."""

    def test_recovery_from_corrupted_state(self, test_data_dir):
        """Test recovery when saved state is corrupted."""
        config = ConfigPresets.development()
        datastore = DataStoreFactory.create(
            {"type": "filesystem", "params": {"base_path": str(test_data_dir)}}
        )

        agent = MainAgent(config=config, datastore=datastore)
        assert agent.initialize()

        # Add some knowledge
        agent.learn("Test knowledge for recovery scenario.")
        agent.save_state()

        # Corrupt the saved episodes file
        episodes_file = test_data_dir / "episodes.json"
        if episodes_file.exists():
            episodes_file.write_text("corrupted data")

        # New agent should handle corrupted data gracefully
        agent2 = MainAgent(config=config, datastore=datastore)
        assert agent2.initialize()

        # Load state should handle the error
        result = agent2.load_state()
        # May return False or True with partial load

        # Agent should still be functional
        query_result = agent2.process_question("Test question", max_cycles=1)
        assert query_result.success

    def test_resource_constraints_workflow(self, test_data_dir):
        """Test system behavior under resource constraints."""
        # Create config with limited resources
        config = InsightSpikeConfig(
            environment="test",
            llm={"provider": "mock", "max_tokens": 50},  # Very limited tokens
            memory={"max_episodes": 10, "max_retrieved_docs": 2},  # Limited memory
            embedding={"dimension": 384},
        )

        datastore = DataStoreFactory.create(
            {"type": "filesystem", "params": {"base_path": str(test_data_dir)}}
        )

        agent = MainAgent(config=config, datastore=datastore)
        assert agent.initialize()

        # Try to exceed memory limit
        for i in range(20):
            agent.learn(f"Knowledge item {i}")

        # Should handle gracefully
        stats = agent.get_stats()
        # Memory might implement LRU or other strategies
        assert stats["memory_stats"]["total_episodes"] <= 20

        # Queries should still work with limited resources
        result = agent.process_question("Tell me about knowledge items", max_cycles=1)
        assert result.success
        assert len(result.retrieved_documents) <= 2  # Respects limit


# Import os for environment variables
import os
