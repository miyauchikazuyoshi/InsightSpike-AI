import numpy as np, pytest, math, os, sys
# Ensure src path
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from indexes.vector_index import InMemoryIndex
try:
    from indexes.hnsw_index import HNSWLibIndex
except Exception:
    HNSWLibIndex = None  # type: ignore

@pytest.mark.skipif(HNSWLibIndex is None, reason="hnswlib not installed")
def test_hnsw_parity_small_random():
    dim=32
    n=200
    rng=np.random.default_rng(42)
    data=rng.normal(size=(n,dim)).astype(np.float32)
    q=rng.normal(size=(dim,)).astype(np.float32)
    ids=list(range(n))
    brute=InMemoryIndex(dim=dim)
    brute.add(ids,data)
    try:
        ann=HNSWLibIndex(dim=dim,max_elements=500)
    except ImportError:
        pytest.skip('hnswlib not available at runtime')
    ann.add(ids,data)
    topk=10
    brute_res=brute.search(q, topk)
    ann_res=ann.search(q, topk)
    # Check that overlap >= 80% (ANN may differ slightly)
    brute_ids={i for i,_ in brute_res}
    ann_ids={i for i,_ in ann_res}
    overlap=len(brute_ids & ann_ids)
    assert overlap >= int(topk*0.8), f"Overlap {overlap} < 80% of {topk}"
    # Distances monotonic ascending
    assert all(ann_res[i][1] <= ann_res[i+1][1] + 1e-9 for i in range(len(ann_res)-1))
