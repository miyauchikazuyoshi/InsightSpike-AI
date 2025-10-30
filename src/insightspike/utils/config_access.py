"""Config access utilities to centralize legacy getattr/hasattr chains.
Lines here are tagged to be ignored by branching analyzer.

Lightweight diagnostics (guarded by INSIGHTSPIKE_DIAG_IMPORT) added to
investigate an import-time hang observed when importing main_agent.
These prints are cheap and can be removed after root cause isolation.
"""
from __future__ import annotations
from typing import Any
import os, sys

if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':  # minimal overhead
    print('[config_access] import start', flush=True)

IGNORE_TAG = 'CONFIG_BRANCH_IGNORE'

def safe_attr(obj: Any, path: str, default: Any=None):  # CONFIG_BRANCH_IGNORE
    cur = obj
    for part in path.split('.'):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur

def safe_has(obj: Any, path: str) -> bool:  # CONFIG_BRANCH_IGNORE
    cur = obj
    for part in path.split('.'):
        if cur is None:
            return False
        if isinstance(cur, dict):
            if part not in cur:
                return False
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return False
            cur = getattr(cur, part)
    return True

__all__ = ['safe_attr', 'safe_has']

if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':  # end marker
    print('[config_access] import end', flush=True)
