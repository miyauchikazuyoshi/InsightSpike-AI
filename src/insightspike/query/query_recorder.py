"""QueryRecorder

Phase1b: 抽出スケルトン
- MainAgent 内のクエリ保存/ID生成/メタ構築ロジックを移譲する予定
- ここでは I/O 抽象 (datastore) と ID 生成, メタ辞書合成の枠のみ実装
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

@dataclass
class QueryRecord:
    id: str
    text: str
    vec: Optional[Any]
    response: str
    has_spike: bool
    spike_episode_id: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]

class QueryRecorder:
    def __init__(self, datastore: Optional[Any]):
        self.datastore = datastore

    def build_record(
        self,
        text: str,
        response: str,
        has_spike: bool,
        spike_episode_id: Optional[str],
        query_vec: Optional[Any],
        processing_time: float,
        total_cycles: int,
        converged: bool,
        reasoning_quality: float,
        retrieved_doc_count: int,
        llm_provider: str,
    ) -> QueryRecord:
        qid = f"query_{int(time.time()*1000)}_{str(uuid.uuid4())[:8]}"
        metadata = {
            'processing_time': processing_time,
            'total_cycles': total_cycles,
            'converged': converged,
            'reasoning_quality': reasoning_quality,
            'retrieved_doc_count': retrieved_doc_count,
            'llm_provider': llm_provider,
        }
        return QueryRecord(
            id=qid,
            text=text,
            vec=query_vec,
            response=response,
            has_spike=has_spike,
            spike_episode_id=spike_episode_id,
            timestamp=time.time(),
            metadata=metadata,
        )

    def save(self, records: Sequence[QueryRecord]) -> bool:
        if not self.datastore or not hasattr(self.datastore, 'save_queries'):
            return False
        payload = [
            {
                'id': r.id,
                'text': r.text,
                'vec': r.vec,
                'has_spike': r.has_spike,
                'spike_episode_id': r.spike_episode_id,
                'response': r.response,
                'timestamp': r.timestamp,
                'metadata': r.metadata,
            }
            for r in records
        ]
        try:
            return bool(self.datastore.save_queries(payload, namespace='queries'))
        except Exception:
            return False

__all__ = ['QueryRecorder', 'QueryRecord']
