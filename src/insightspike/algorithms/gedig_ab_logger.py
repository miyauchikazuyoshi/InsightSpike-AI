"""geDIG A/B Comparison Logger (Phase3→Phase4 拡張)

機能:
 1. pure / full 両モードの geDIG (gedig, ged, ig) を同一クエリで取得し rolling Pearson 相関を算出
 2. 相関が threshold を下回った際 (min_pairs 以上) に WARNING + alert CSV 追記
 3. flush_every 件追加ごとに全履歴 CSV をオートフラッシュ
 4. CSV 列: query_id, pure/full gedig/ged/ig, k_estimate, window_corr_at_record, timestamp
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
from collections import deque
import math
import time
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class GeDIGResult:
    mode: str
    gedig: float
    ged: Optional[float]
    ig: Optional[float]


class GeDIGABLogger:
    def __init__(self, window: int = 100, threshold: float = 0.85, min_pairs: int = 30, flush_every: int = 50) -> None:
        self.window = window
        self.threshold = threshold
        self.min_pairs = min_pairs
        self.flush_every = flush_every
        self._pairs: deque = deque(maxlen=window)
        self._history: List[Tuple[str, GeDIGResult, GeDIGResult, float, float]] = []
        self._last_alert_count = -1
        self._auto_csv_path: Optional[str] = None
        self._last_flush_len = 0
        self._alert_events: List[Tuple[int, float, float]] = []
        self._alert_csv_path: Optional[str] = None
        self._writer = None  # Optional injected writer (file-like), used in export if set

    # ---------------- internal utils ----------------
    def _pearson(self, xs, ys) -> float:
        n = len(xs)
        if n < 2:
            return float('nan')
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x-mx) ** 2 for x in xs))
        den_y = math.sqrt(sum((y-my) ** 2 for y in ys))
        # Zero variance => perfectly correlated (all identical values).
        # Returning 1.0 stabilizes early window metrics and avoids NaN propagation in stats scripts.
        if den_x == 0 or den_y == 0:
            return 1.0
        return num / (den_x * den_y)

    def _derive_k(self, res: GeDIGResult) -> tuple[Optional[float], Optional[str]]:
        """Return (k_estimate, missing_reason).

        missing_reason codes:
          - ged_missing: ged is None
          - ig_missing: ig is None
          - ig_zero: ig == 0 (avoid division)
          - non_finite: computed k is not finite
          - calc_error: unexpected exception
          - None: k successfully computed
        """
        try:
            if res.ged is None:
                return None, 'ged_missing'
            if res.ig is None:
                return None, 'ig_missing'
            if res.ig == 0:
                return None, 'ig_zero'
            k = (res.ged - res.gedig) / res.ig
            if not math.isfinite(k):
                return None, 'non_finite'
            return k, None
        except Exception:  # pragma: no cover
            return None, 'calc_error'

    # ---------------- public API ----------------
    def set_auto_csv_path(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        except Exception:  # pragma: no cover
            logger.debug("Failed to ensure directory for auto csv path")
        self._auto_csv_path = path
        b, e = os.path.splitext(path)
        self._alert_csv_path = f"{b}_alerts{e or '.csv'}"

    def set_writer(self, writer: Any) -> None:
        """Inject a file-like writer (supports .write) for export.

        When set, export_csv will use this writer instead of opening a file.
        """
        self._writer = writer

    def current_metrics(self) -> Dict[str, Any]:
        if not self._pairs:
            return {'count': 0}
        pv = [p[0].gedig for p in self._pairs]
        fv = [p[1].gedig for p in self._pairs]
        corr = self._pearson(pv, fv)
        return {
            'count': len(self._pairs),
            'gedig_corr': corr,
            'window': self.window,
            'threshold': self.threshold,
            'min_pairs': self.min_pairs,
        }

    def record(self, query_id: str, pure_res: Dict[str, Any], full_res: Dict[str, Any]) -> None:
        pure_obj = GeDIGResult('pure', float(pure_res.get('gedig', 0.0)), pure_res.get('ged'), pure_res.get('ig'))
        full_obj = GeDIGResult('full', float(full_res.get('gedig', 0.0)), full_res.get('ged'), full_res.get('ig'))
        self._pairs.append((pure_obj, full_obj))
        metrics = self.current_metrics()
        wc = float(metrics.get('gedig_corr')) if 'gedig_corr' in metrics else float('nan')
        self._history.append((query_id, pure_obj, full_obj, time.time(), wc))
        try:
            cnt = metrics.get('count', 0)
            corr = metrics.get('gedig_corr')
            if (
                cnt >= self.min_pairs and isinstance(corr, (float, int)) and not math.isnan(corr)
                and corr < self.threshold and cnt != self._last_alert_count
            ):
                logger.warning(
                    f"geDIG A/B correlation {corr:.3f} below threshold {self.threshold:.2f} (n={cnt})"
                )
                self._last_alert_count = cnt
                self._alert_events.append((cnt, float(corr), float(self.threshold)))
                if self._alert_csv_path:
                    try:
                        first = not os.path.exists(self._alert_csv_path)
                        import csv
                        with open(self._alert_csv_path, 'a', newline='') as af:
                            w = csv.writer(af)
                            if first:
                                w.writerow(['count', 'gedig_corr', 'threshold'])
                            w.writerow([cnt, f"{corr:.6f}", f"{self.threshold:.6f}"])
                    except Exception:  # pragma: no cover
                        logger.debug("Failed writing alert CSV")
            if self._auto_csv_path and len(self._history) - self._last_flush_len >= self.flush_every:
                written = self.export_csv(self._auto_csv_path)
                self._last_flush_len = len(self._history)
                logger.info(f"geDIG A/B auto-flush CSV ({written} rows) -> {self._auto_csv_path}")
        except Exception as e:  # pragma: no cover
            logger.debug(f"A/B logger side-effect failed: {e}")

    def export_csv(self, path: str) -> int:
        import csv
        rows = 0
        if self._writer is not None:
            w = csv.writer(self._writer)
            w.writerow([
                'query_id','pure_gedig','full_gedig','pure_ged','full_ged','pure_ig','full_ig','k_estimate','k_missing_reason','window_corr_at_record','timestamp'
            ])
            for qid, p, f_res, ts, wc in self._history:
                k_val, k_reason = self._derive_k(p)
                w.writerow([
                    qid,
                    p.gedig, f_res.gedig,
                    p.ged, f_res.ged,
                    p.ig, f_res.ig,
                    k_val,
                    k_reason or '',
                    wc,
                    f"{ts:.6f}"
                ])
                rows += 1
            return rows + 1
        # Fallback: write to file path
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'query_id','pure_gedig','full_gedig','pure_ged','full_ged','pure_ig','full_ig','k_estimate','k_missing_reason','window_corr_at_record','timestamp'
            ])
            for qid, p, f_res, ts, wc in self._history:
                k_val, k_reason = self._derive_k(p)
                w.writerow([
                    qid,
                    p.gedig, f_res.gedig,
                    p.ged, f_res.ged,
                    p.ig, f_res.ig,
                    k_val,
                    k_reason or '',
                    wc,
                    f"{ts:.6f}"
                ])
                rows += 1
        return rows + 1

__all__ = ['GeDIGABLogger']
