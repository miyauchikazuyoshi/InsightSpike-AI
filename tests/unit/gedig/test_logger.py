"""Unit tests for gedig.logger module."""

import csv
import gzip
import os
import tempfile

import pytest

from insightspike.algorithms.gedig.logger import GeDIGLogger
from insightspike.algorithms.gedig.types import GeDIGResult


class TestGeDIGLogger:
    """Tests for GeDIGLogger class."""

    def test_init_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path)
            assert os.path.exists(logger._current_file)
            logger.close()

    def test_init_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path)
            assert logger.max_lines == 50_000
            assert logger.max_bytes == 50 * 1024 * 1024
            assert logger.compress_on_rotate is False
            logger.close()

    def test_init_custom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(
                output_path=path,
                max_lines=100,
                max_bytes=1024,
                compress_on_rotate=True,
            )
            assert logger.max_lines == 100
            assert logger.max_bytes == 1024
            assert logger.compress_on_rotate is True
            logger.close()

    def test_log_single_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path)
            result = GeDIGResult(
                gedig_value=-0.5,
                ged_value=0.3,
                ig_value=0.8,
                structural_improvement=0.2,
                ig_z_score=0.5,
                hop0_reward=0.1,
                aggregate_reward=0.15,
                reward=0.12,
                spike=True,
            )
            logger.log(step=0, result=result)
            logger.close()

            # Read and verify
            with open(logger._current_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["step"] == "0"
            assert rows[0]["spike"] == "1"

    def test_log_multiple_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path)
            for i in range(10):
                result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8)
                logger.log(step=i, result=result)
            logger.close()

            with open(logger._current_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 10

    def test_build_filename_with_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path)
            assert logger._build_filename().endswith("_0.csv")
            logger.close()

    def test_build_filename_without_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test")
            logger = GeDIGLogger(output_path=path)
            assert logger._build_filename().endswith("_0.csv")
            logger.close()

    def test_rotation_by_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path, max_lines=5)
            for i in range(12):
                result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8)
                logger.log(step=i, result=result)
            logger.close()

            # Should have rotated at least once
            assert logger._file_index >= 1
            file0 = os.path.join(tmpdir, "test_0.csv")
            file1 = os.path.join(tmpdir, "test_1.csv")
            assert os.path.exists(file0)
            assert os.path.exists(file1)

    def test_rotation_with_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path, max_lines=5, compress_on_rotate=True)
            for i in range(12):
                result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8)
                logger.log(step=i, result=result)
            logger.close()

            # First file should be compressed
            gzip_file = os.path.join(tmpdir, "test_0.csv.gz")
            assert os.path.exists(gzip_file)

            # Verify gzip file is readable
            with gzip.open(gzip_file, "rt") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 5

    def test_pathlib_path_support(self):
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            logger = GeDIGLogger(output_path=path)
            result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8)
            logger.log(step=0, result=result)
            logger.close()
            assert os.path.exists(logger._current_file)

    def test_fields_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path)
            expected_fields = [
                "step",
                "raw_ged",
                "ged_value",
                "structural_improvement",
                "ig_raw",
                "ig_z_score",
                "hop0_reward",
                "aggregate_reward",
                "reward",
                "spike",
                "version",
            ]
            assert logger.fields == expected_fields
            logger.close()

    def test_header_written_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            logger = GeDIGLogger(output_path=path)
            result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8)
            logger.log(step=0, result=result)
            logger.log(step=1, result=result)
            logger.close()

            with open(logger._current_file) as f:
                lines = f.readlines()
            # Header + 2 data lines
            assert len(lines) == 3
            assert "step" in lines[0]  # Header

    def test_rotation_by_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            # Very small max_bytes to trigger rotation
            logger = GeDIGLogger(output_path=path, max_lines=10000, max_bytes=200)
            for i in range(50):
                result = GeDIGResult(
                    gedig_value=-0.5,
                    ged_value=0.3,
                    ig_value=0.8,
                    structural_improvement=0.12345678,
                )
                logger.log(step=i, result=result)
            logger.close()

            # Should have rotated due to size
            assert logger._file_index >= 1
