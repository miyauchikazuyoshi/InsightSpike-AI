"""InsightSpike package metadata (light-mode aware).

このモジュールは従来、多数の下位モジュールをトップレベル import していましたが
`INSIGHTSPIKE_LITE_MODE=1` あるいは `INSIGHTSPIKE_MIN_IMPORT=1` が設定された
軽量モードでは **一切の重い依存 import を行わない** ように簡略化しました。

目的:
 - selftest / 部分機能 (例: `insightspike.algorithms.gedig_ab_logger`) の高速 & 安全 import
 - PyTorch / 外部 API / DB 初期化によるハングを回避

Lite モード判定:
 - INSIGHTSPIKE_LITE_MODE=1
 - もしくは INSIGHTSPIKE_MIN_IMPORT=1 (後方互換)

非 Lite モードでは従来どおり可能な範囲で従来 API を再エクスポートします。
ImportError は握りつぶし、None / プレースホルダを提供します。
"""
from __future__ import annotations
import os


class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.8.0"


# ---- Mode Detection -------------------------------------------------------
LITE_MODE = (
    os.environ.get("INSIGHTSPIKE_LITE_MODE", "0") == "1"
    or os.environ.get("INSIGHTSPIKE_MIN_IMPORT", "0") == "1"
)

# ---- Placeholders (always defined to keep attribute existence stable) -----
class MainAgent:
    def __init__(self, *_, **__):
        pass
    def initialize(self):
        return False
    def process_question(self, question, **kwargs):  # pragma: no cover - trivial
        return {"response": "MainAgent not available (lite mode)" if LITE_MODE else "MainAgent placeholder", "success": False}

class CycleResult:
    def __init__(self, **kwargs):
        pass

ErrorMonitor = None
L2MemoryManager = None
get_llm_provider = None
create_agent = None
quick_demo = None
L3GraphReasoner = None
GenericInsightSpikeAgent = None
InsightSpikeAgentFactory = None
create_maze_agent = None
create_configured_maze_agent = None
AgentConfigBuilder = None
TaskType = None
EnvironmentInterface = None
InsightMoment = None
StandaloneL3GraphReasoner = None
create_standalone_reasoner = None
analyze_documents_simple = None

# config や utils は Lite モードでは遅延 import (実際に必要になるまで読み込まない)
def _lazy_get_config():  # pragma: no cover (軽量経路でのダミー)
    return {}

get_config = _lazy_get_config  # 先にシンボルだけ定義
graph_metrics = None
eureka_spike = None


if not LITE_MODE:
    # 重い import 群。失敗しても例外を伝播させない。
    try:
        from .config import get_config as _real_get_config
        # Config 型も可能なら取得
        try:  # ネストして Config だけ失敗しても get_config は維持
            from .config import Config as _Cfg  # type: ignore
            Config = _Cfg  # type: ignore
        except Exception:  # pragma: no cover
            Config = None  # type: ignore
        get_config = _real_get_config  # type: ignore
    except Exception:  # pragma: no cover
        pass
    # Defer importing main_agent until attribute actually accessed to
    # break potential circular import during debug (pytest collection hang).
    # PEP562 __getattr__ below will load lazily.
    try:
        from .implementations.layers.layer1_error_monitor import ErrorMonitor as _Err
        from .implementations.layers.layer2_memory_manager import L2MemoryManager as _L2
        from .implementations.layers.layer4_llm_interface import get_llm_provider as _prov
        from .quick_start import create_agent as _create_agent, quick_demo as _quick_demo
        ErrorMonitor = _Err  # type: ignore
        L2MemoryManager = _L2  # type: ignore
        get_llm_provider = _prov  # type: ignore
        create_agent = _create_agent  # type: ignore
        quick_demo = _quick_demo  # type: ignore
    except Exception:  # pragma: no cover
        pass
    try:
        from .implementations.layers.layer3_graph_reasoner import L3GraphReasoner as _L3
        L3GraphReasoner = _L3  # type: ignore
    except Exception:  # pragma: no cover
        pass
    try:
        from .core.base.generic_interfaces import (
            EnvironmentInterface as _Env,
            InsightMoment as _IM,
            TaskType as _TT,
        )
        from .implementations.agents.agent_factory import (
            AgentConfigBuilder as _ACB,
            InsightSpikeAgentFactory as _ASF,
            create_configured_maze_agent as _cc_maze,
            create_maze_agent as _c_maze,
        )
        from .implementations.agents.generic_agent import GenericInsightSpikeAgent as _Gen
        EnvironmentInterface = _Env  # type: ignore
        InsightMoment = _IM  # type: ignore
        TaskType = _TT  # type: ignore
        AgentConfigBuilder = _ACB  # type: ignore
        InsightSpikeAgentFactory = _ASF  # type: ignore
        create_configured_maze_agent = _cc_maze  # type: ignore
        create_maze_agent = _c_maze  # type: ignore
        GenericInsightSpikeAgent = _Gen  # type: ignore
    except Exception:  # pragma: no cover
        pass
    try:
        from .tools.standalone.standalone_l3 import (
            StandaloneL3GraphReasoner as _SL3,
            analyze_documents_simple as _ads,
            create_standalone_reasoner as _csr,
        )
        StandaloneL3GraphReasoner = _SL3  # type: ignore
        analyze_documents_simple = _ads  # type: ignore
        create_standalone_reasoner = _csr  # type: ignore
    except Exception:  # pragma: no cover
        pass
    try:
        from .detection import eureka_spike as _eureka
        eureka_spike = _eureka  # type: ignore
    except Exception:  # pragma: no cover
        pass
    try:
        from .metrics import graph_metrics as _gm
        graph_metrics = _gm  # type: ignore
    except Exception:  # pragma: no cover
        pass
    try:
        from . import utils  # noqa: F401  (re-export purpose)
    except Exception:  # pragma: no cover
        pass


__version__ = About.VERSION


def about():  # pragma: no cover - trivial accessor
    """Return package information as a dictionary."""
    return {"name": About.NAME, "version": About.VERSION, "lite_mode": LITE_MODE}

def __getattr__(name):  # PEP 562 lazy attribute access for heavy imports
    if name in {"MainAgent", "CycleResult"} and not LITE_MODE:
        try:
            from .implementations.agents.main_agent import MainAgent as _MA, CycleResult as _CR
            globals()["MainAgent"] = _MA  # cache
            globals()["CycleResult"] = _CR
            return _MA if name == "MainAgent" else _CR
        except Exception as e:  # pragma: no cover
            raise AttributeError(f"Could not load {name}: {e}")
    raise AttributeError(name)


__all__ = [
    # Core meta
    "MainAgent",
    "CycleResult",
    "get_config",
    "Config",
    "About",
    "about",
    # Optional graph / detection (may be None in lite)
    "graph_metrics",
    "eureka_spike",
    # Layer exports (may be None)
    "ErrorMonitor",
    "L2MemoryManager",
    "get_llm_provider",
    "L3GraphReasoner",
    # Generic agent system exports (may be None)
    "GenericInsightSpikeAgent",
    "InsightSpikeAgentFactory",
    "create_maze_agent",
    "create_configured_maze_agent",
    "AgentConfigBuilder",
    "TaskType",
    "EnvironmentInterface",
    "InsightMoment",
    # Standalone reasoner exports (may be None)
    "StandaloneL3GraphReasoner",
    "create_standalone_reasoner",
    "analyze_documents_simple",
]
