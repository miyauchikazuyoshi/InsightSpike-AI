import numpy as np
import sys, os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from indexes.datastore_index import DataStoreIndex  # type: ignore

def test_datastore_index_add_search_remove():
    idx = DataStoreIndex(datastore=None)  # in-memory fallback
    vecs = np.random.rand(5, 8).astype(float)
    ids = list(range(10, 15))
    idx.add(ids, vecs)
    assert len(idx) == 5
    q = vecs[0]
    res = idx.search(q, top_k=3)
    assert res and res[0][0] in ids
    # remove two ids
    idx.remove(ids[:2])
    assert len(idx) == 3
    # search still returns remaining ids only
    res2 = idx.search(q, top_k=5)
    remaining = {r[0] for r in res2}
    assert not any(x in remaining for x in ids[:2])
