"""Preset loader for maze-navigation-enhanced experiments.

Precedence: ENV < preset (YAML) < CLI overrides

Usage:
    from utils.preset_loader import load_preset, apply_env
    cfg = load_preset(preset_name='25x25', overrides={'gedig': {'threshold': -0.14}})
    apply_env(cfg)  # export selected keys to os.environ for legacy scripts
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / 'configs'


def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data


def _env_to_cfg() -> Dict[str, Any]:
    """Collect known MAZE_* env vars under cfg['env'] for reference.

    These are low-priority defaults in this loader's precedence.
    """
    env_keys = [k for k in os.environ.keys() if k.startswith('MAZE_')]
    env_map = {k: os.environ.get(k) for k in env_keys}
    return {'env': env_map}


def resolve_preset_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if n in ('15', '15x15', 'size15'):
        return '15x15'
    if n in ('25', '25x25', 'size25'):
        return '25x25'
    if n in ('50', '50x50', 'size50'):
        return '50x50'
    return n


def load_preset(*, preset_name: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None,
                default_name: str = 'default') -> Dict[str, Any]:
    """Load merged configuration.

    Order: env (low) <- default.yaml <- preset.yaml <- overrides (high)
    """
    cfg = _env_to_cfg()
    # default.yaml
    default_yaml = _load_yaml(CONF_DIR / f'{default_name}.yaml')
    cfg = _deep_merge(cfg, default_yaml)
    # preset
    pname = resolve_preset_name(preset_name)
    if pname:
        # map logical name to file
        fname = {
            '15x15': '15x15.yaml',
            '25x25': '25x25.yaml',
            '50x50': '50x50.yaml',
        }.get(pname, f'{pname}.yaml')
        preset_path = CONF_DIR / fname
        cfg = _deep_merge(cfg, _load_yaml(preset_path))
    # CLI overrides
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def apply_env(cfg: Dict[str, Any]) -> None:
    """Export selected config fields to environment for legacy scripts.

    - Writes cfg['env'] keys as strings
    - Also publishes convenience variables (size, factors) if present
    """
    env = cfg.get('env', {}) or {}
    for k, v in env.items():
        try:
            os.environ[str(k)] = v if isinstance(v, str) else json.dumps(v)
        except Exception:
            pass
    # Convenience exports
    maze = (cfg.get('maze') or {})
    if 'max_steps_factor' in maze:
        os.environ['MAZE_MAX_STEPS_FACTOR'] = str(maze['max_steps_factor'])


__all__ = ['load_preset', 'apply_env']

