"""Coverage for path_utils helpers."""

from __future__ import annotations

from pathlib import Path

from insightspike.utils import path_utils


def test_resolve_project_relative_handles_relative_and_abs(tmp_path):
    rel_path = "some/dir/file.txt"
    resolved_rel = Path(path_utils.resolve_project_relative(rel_path))
    assert resolved_rel.is_absolute()
    abs_path = path_utils.resolve_project_relative("/tmp")
    assert Path(abs_path) == Path("/tmp")
