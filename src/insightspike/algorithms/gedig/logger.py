"""CSV logging for geDIG results.

This module provides rotating CSV logging for geDIG calculation results.
"""

from __future__ import annotations

import csv
import gzip
import logging
import os
import shutil
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from .types import GeDIGResult

logger = logging.getLogger(__name__)


class GeDIGLogger:
    """CSV logger with rotation by line count or file size.

    Parameters
    ----------
    output_path
        Base file path (extension optional). Supports pathlib.Path.
    max_lines
        Maximum data lines per file (header excluded).
    max_bytes
        Maximum file size in bytes.
    compress_on_rotate
        Whether to gzip rotated files.
    """

    def __init__(
        self,
        output_path: Any,
        max_lines: int = 50_000,
        max_bytes: int = 50 * 1024 * 1024,
        compress_on_rotate: bool = False,
    ) -> None:
        # PathLike 対応: 早期に str へ正規化
        self.output_path = str(output_path)
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.compress_on_rotate = compress_on_rotate
        self._line_count = 0
        self._file_index = 0
        self.fields: List[str] = [
            'step',
            'raw_ged',
            'ged_value',
            'structural_improvement',
            'ig_raw',
            'ig_z_score',
            'hop0_reward',
            'aggregate_reward',
            'reward',
            'spike',
            'version',
        ]
        self._fh: Any = None
        self._writer: Any = None
        self._current_file: str = ""
        self._open_writer()

    def _rotate_needed(self) -> bool:
        """Check if rotation is needed."""
        try:
            size = os.path.getsize(self._current_file)
        except OSError:
            size = 0
        return self._line_count >= self.max_lines or size >= self.max_bytes

    def _open_writer(self) -> None:
        """Open new log file for writing."""
        self._current_file = self._build_filename()
        first = not os.path.exists(self._current_file)
        self._fh = open(self._current_file, 'a', newline='')
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
        if first:
            self._writer.writeheader()
            self._fh.flush()
            self._line_count = 0

    def _build_filename(self) -> str:
        """Build filename with index."""
        base = self.output_path
        if '.' in base and not base.endswith('.'):
            root, ext = base.rsplit('.', 1)
        else:
            root, ext = base, 'csv'
        return f"{root}_{self._file_index}.{ext}"

    def log(self, step: int, result: 'GeDIGResult') -> None:
        """Log a geDIG result."""
        if self._rotate_needed():
            old_file = self._current_file
            self._fh.close()
            if self.compress_on_rotate:
                try:
                    with open(old_file, 'rb') as f_in:
                        with gzip.open(old_file + '.gz', 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(old_file)
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Compression failed for {old_file}: {e}")
            self._file_index += 1
            self._open_writer()

        row = {
            'step': step,
            'raw_ged': result.raw_ged,
            'ged_value': result.ged_value,
            'structural_improvement': result.structural_improvement,
            'ig_raw': result.ig_raw,
            'ig_z_score': result.ig_z_score,
            'hop0_reward': result.hop0_reward,
            'aggregate_reward': result.aggregate_reward,
            'reward': result.reward,
            'spike': int(result.has_spike),
            'version': result.version,
        }
        self._writer.writerow(row)
        self._fh.flush()
        self._line_count += 1

    def close(self) -> None:  # pragma: no cover
        """Close the log file."""
        try:
            self._fh.close()
        except Exception:
            pass


__all__ = ["GeDIGLogger"]
