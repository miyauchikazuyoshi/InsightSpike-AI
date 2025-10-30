#!/usr/bin/env python3
"""ANN ベンチマーク: InMemoryIndex vs HNSWLibIndex

計測項目:
  - 検索平均/ p95 レイテンシ (ms)
  - 近傍オーバーラップ率 (top_k一致割合)
  - メモリフットプリント概算 (要素数 * dim * 4 bytes)

Usage:
  python -m experiments.ann_benchmark --n 8000 --dim 8 --queries 200 --top_k 8
"""
from __future__ import annotations
import argparse, time, math, statistics as stats, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

try:
    from indexes.vector_index import InMemoryIndex  # type: ignore
except Exception:
    InMemoryIndex = None  # type: ignore
try:
    from indexes.hnsw_index import HNSWLibIndex  # type: ignore
except Exception:
    HNSWLibIndex = None  # type: ignore


def build_data(n:int, dim:int, seed:int=42):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim)).astype('float32')
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-9
    return base

def linear_reference_search(vectors: np.ndarray, queries: np.ndarray, top_k:int):
    # brute force L2
    ref = []
    for q in queries:
        d = np.linalg.norm(vectors - q, axis=1)
        idx = np.argsort(d)[:top_k]
        ref.append(idx.tolist())
    return ref

def benchmark_index(name:str, index, queries: np.ndarray, top_k:int):
    lat=[]
    res=[]
    for q in queries:
        t0=time.perf_counter()
        out = index.search(q, top_k)
        lat.append((time.perf_counter()-t0)*1000.0)
        res.append([i for i,_d in out])
    return {
        'name': name,
        'lat_mean_ms': float(stats.mean(lat)) if lat else 0.0,
        'lat_p95_ms': float(np.percentile(lat,95)) if len(lat)>1 else (lat[0] if lat else 0.0),
        'results': res
    }

def overlap(ref, test):
    # 平均 Jaccard (集合) と 平均ヒット率 (ref内存在率)
    jacc=[]; hit=[]
    for r,t in zip(ref,test):
        rs=set(r); ts=set(t)
        inter=len(rs & ts); union=len(rs|ts) or 1
        jacc.append(inter/union)
        hit.append(inter/len(rs) if rs else 0.0)
    return float(stats.mean(jacc)), float(stats.mean(hit))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--dim', type=int, default=8)
    ap.add_argument('--queries', type=int, default=1200)
    ap.add_argument('--top_k', type=int, default=8)
    ap.add_argument('--ann_m', type=int, default=16)
    ap.add_argument('--ann_ef_construction', type=int, default=100)
    ap.add_argument('--ann_ef_search', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()

    vecs = build_data(args.n, args.dim, args.seed)
    # queries: ランダムサンプル
    q_ids = np.random.default_rng(args.seed+1).integers(0,args.n,size=args.queries)
    queries = vecs[q_ids]

    # Linear reference (brute force) for overlap baseline
    print(f"[build] generating brute force reference for {len(queries)} queries...")
    ref = linear_reference_search(vecs, queries, args.top_k)

    # InMemoryIndex (if available)
    if InMemoryIndex is None:
        print('[warn] InMemoryIndex unavailable')
        return
    lin = InMemoryIndex()
    for i,v in enumerate(vecs):
        lin.add([i], v.reshape(1,-1))
    lin_bench = benchmark_index('linear_index', lin, queries, args.top_k)

    # ANN (HNSW)
    ann_bench = None
    if HNSWLibIndex is None:
        print('[warn] HNSWLibIndex unavailable (install hnswlib)')
    else:
        try:
            ann = HNSWLibIndex(dim=args.dim, max_elements=args.n, M=args.ann_m, ef_construction=args.ann_ef_construction, ef_search=args.ann_ef_search)
            ann.add(list(range(args.n)), vecs)
            ann_bench = benchmark_index('hnsw', ann, queries, args.top_k)
        except Exception as e:
            print('[warn] Failed to init HNSW:', e)

    # Overlap metrics
    lin_j, lin_hit = overlap(ref, lin_bench['results'])
    print(f"[linear] mean_ms={lin_bench['lat_mean_ms']:.4f} p95_ms={lin_bench['lat_p95_ms']:.4f} jacc={lin_j:.3f} hit={lin_hit:.3f}")
    if ann_bench:
        ann_j, ann_hit = overlap(ref, ann_bench['results'])
        print(f"[hnsw]   mean_ms={ann_bench['lat_mean_ms']:.4f} p95_ms={ann_bench['lat_p95_ms']:.4f} jacc={ann_j:.3f} hit={ann_hit:.3f}")
        speedup = lin_bench['lat_mean_ms']/ann_bench['lat_mean_ms'] if ann_bench['lat_mean_ms']>0 else float('inf')
        print(f"[speedup] linear/ann mean={speedup:.2f}x")

if __name__=='__main__':
    main()
