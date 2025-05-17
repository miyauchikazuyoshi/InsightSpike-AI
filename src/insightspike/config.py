from pathlib import Path
from datetime import datetime

# ─── Paths ──────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent.parent    # repo root
DATA_DIR  = ROOT_DIR / "data" / "raw"
LOG_DIR   = ROOT_DIR / "data" / "logs"
INDEX_FILE = ROOT_DIR / "data" / "index.faiss"
GRAPH_FILE = ROOT_DIR / "data" / "graph_pyg.pt"

# ─── Models ─────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME         = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# ─── Retrieval & Graph ─────────────────────────────────
SIM_THRESHOLD = 0.35
TOP_K         = 5

# ─── Eureka Spike thresholds ───────────────────────────
SPIKE_GED  = 0.5
SPIKE_IG   = 0.2
ETA_SPIKE  = 0.2

# ─── Helper ────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)

def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# ─── Memory-edit thresholds ─────────────────────────────
MERGE_GED   = 0.4   # これ以下に ΔGED が下がったら統合候補
SPLIT_IG    = -0.15 # これより ΔIG がマイナスなら分裂候補
PRUNE_C     = 0.05  # C値がこれ未満なら削除候補
INACTIVE_N  = 30    # ループ n 回アクセス無しで削除
