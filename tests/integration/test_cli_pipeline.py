#!/usr/bin/env python3
"""
CLI Pipeline Integration Tests
==============================

Test the CLI commands work correctly end-to-end.
"""

import pytest
import subprocess
import tempfile
from pathlib import Path
import json


class TestCLIPipeline:
    """Test CLI commands in realistic scenarios"""
    
    def run_cli_command(self, command: list) -> dict:
        """Run a CLI command and return result"""
        result = subprocess.run(
            ["poetry", "run", "spike"] + command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
    def test_help_command(self):
        """Test help command displays correctly"""
        result = self.run_cli_command(["--help"])
        
        assert result["success"]
        assert "InsightSpike AI" in result["stdout"]
        assert "Commands" in result["stdout"]
        
    def test_version_command(self):
        """Test version command"""
        result = self.run_cli_command(["version"])
        
        assert result["success"]
        assert "InsightSpike AI" in result["stdout"]
        assert "0.8.0" in result["stdout"]
        
    def test_embed_and_query_pipeline(self):
        """Test embedding documents and querying"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test document
            test_file = Path(tmpdir) / "knowledge.txt"
            test_file.write_text("""
            Quantum mechanics describes the behavior of matter at atomic scales.
            It introduces concepts like superposition and entanglement.
            """)
            
            # Embed the document
            embed_result = self.run_cli_command(["embed", str(test_file)])
            assert embed_result["success"]
            assert "Documents added successfully" in embed_result["stdout"]
            
            # Query about the content
            query_result = self.run_cli_command(["query", "What is quantum mechanics?"])
            assert query_result["success"]
            assert "quantum" in query_result["stdout"].lower() or \
                   "atomic" in query_result["stdout"].lower()
                   
    def test_stats_command(self):
        """Test stats command shows statistics"""
        result = self.run_cli_command(["stats"])
        
        assert result["success"]
        assert "Agent Statistics" in result["stdout"]
        
    def test_command_aliases(self):
        """Test command aliases work"""
        # Test 'q' alias for query
        result_q = self.run_cli_command(["q", "Test question"])
        assert result_q["success"]
        
        # Test 'ask' alias for query
        result_ask = self.run_cli_command(["ask", "Test question"])
        assert result_ask["success"]
        
    def test_multiple_file_embedding(self):
        """Test embedding multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(3):
                file_path = Path(tmpdir) / f"doc_{i}.txt"
                file_path.write_text(f"Document {i} content")
                
            # Embed entire directory
            result = self.run_cli_command(["embed", tmpdir])
            assert result["success"]
            assert "3 files" in result["stdout"]
            
    def test_verbose_mode(self):
        """Test verbose output"""
        result = self.run_cli_command(["query", "Test", "--verbose"])
        
        assert result["success"]
        assert "Quality score" in result["stdout"] or "Retrieved" in result["stdout"]
        
    def test_different_presets(self):
        """Test different configuration presets"""
        # Development preset
        dev_result = self.run_cli_command(["query", "Test", "--preset", "development"])
        assert dev_result["success"]
        
        # Experiment preset
        exp_result = self.run_cli_command(["query", "Test", "--preset", "experiment"])
        assert exp_result["success"]
        
    def test_error_handling(self):
        """Test CLI handles errors gracefully"""
        # Non-existent file
        result = self.run_cli_command(["embed", "/nonexistent/file.txt"])
        assert not result["success"]
        assert "not found" in result["stdout"].lower()
        
    @pytest.mark.slow
    def test_large_document_processing(self):
        """Test processing larger documents"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a larger document
            large_doc = Path(tmpdir) / "large.txt"
            content = "\n".join([f"Line {i}: " + "x" * 100 for i in range(100)])
            large_doc.write_text(content)
            
            # Embed
            result = self.run_cli_command(["embed", str(large_doc)])
            assert result["success"]
            
    def test_markdown_file_support(self):
        """Test markdown file embedding"""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "test.md"
            md_file.write_text("# Test\n\nThis is a **markdown** file.")
            
            result = self.run_cli_command(["embed", str(md_file)])
            assert result["success"]