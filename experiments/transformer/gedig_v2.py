#!/usr/bin/env python3
"""
geDIG v2: Transformer対応版

統一geDIG公式:
    F = ΔEPC - λ(ΔH + γΔSP)

成分:
    - ΔEPC: 構造変化コスト（有意なattention変化のみ）
    - ΔH: エントロピー変化（attention分布の集中度）
    - ΔSP: ショートカット純度変化（アンカーへの経路集中度）

迷路との対応:
    - 迷路: 疎→密（エッジ追加でショートカット発見）
    - Transformer: 密→疎（プルーニングでショートカット残存）
    - 両方: 最終的に効率的なショートカットが存在する状態を目指す
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GeDIGv2Result:
    """geDIG v2 計算結果"""
    F: torch.Tensor  # geDIG値（低い=良い）
    F_mean: float
    delta_epc: float
    delta_h: float
    delta_sp: float
    h_before: float
    h_after: float
    sp_before: float
    sp_after: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "F_mean": self.F_mean,
            "delta_epc": self.delta_epc,
            "delta_h": self.delta_h,
            "delta_sp": self.delta_sp,
            "h_before": self.h_before,
            "h_after": self.h_after,
            "sp_before": self.sp_before,
            "sp_after": self.sp_after,
        }


class GeDIGv2(nn.Module):
    """
    geDIG v2: Transformer対応版

    F = ΔEPC - λ(ΔH + γΔSP)

    - ΔEPC: 構造変化コスト（有意な変化のみ）
    - ΔH: エントロピー変化（延伸利得）
    - ΔSP: ショートカット純度変化（CLSへの経路集中度）

    基本原則: 延伸（ΔH>0）が利得
    - ΔH > 0: 探索拡大 = 情報利得 → F低下（良い）
    - ΔSP > 0: ショートカット形成 → F低下（良い）

    アブレーション用: entropy_sign=-1 で集中化を利得とする実験可能
    """

    def __init__(
        self,
        lambda_param: float = 1.0,
        gamma: float = 0.5,
        epc_threshold: float = 0.05,
        sp_k_ratio: float = 0.2,
        anchor_indices: Optional[List[int]] = None,
        entropy_sign: int = 1,  # デフォルト: 延伸が利得（geDIG基本原則）
    ):
        """
        Args:
            lambda_param: 情報利得項の重み
            gamma: SP項の重み（ΔH + γΔSP）
            epc_threshold: EPCで「有意な変化」とみなす閾値
            sp_k_ratio: 上位何%を「ショートカット」とみなすか
            anchor_indices: アンカートークンの位置 [0]=CLS, [-1]=SEP
            entropy_sign: エントロピー項の符号
                1: 延伸モード（ΔH>0が良い、デフォルト = geDIG基本原則）
               -1: 集中モード（ΔH<0が良い、アブレーション用）
        """
        super().__init__()
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.epc_threshold = epc_threshold
        self.sp_k_ratio = sp_k_ratio
        self.anchor_indices = anchor_indices if anchor_indices is not None else [0]
        self.entropy_sign = entropy_sign

    def forward(
        self,
        attn_before: torch.Tensor,
        attn_after: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> GeDIGv2Result:
        """
        geDIG v2 を計算

        Args:
            attn_before: (B, H, S, S) 参照状態のattention
            attn_after: (B, H, S, S) 現在のattention
            mask: (B, S) padding mask (1=valid, 0=padding)

        Returns:
            GeDIGv2Result with F, delta_epc, delta_h, delta_sp
        """
        # 1. ΔEPC: 有意な構造変化
        delta_epc = self._compute_epc(attn_before, attn_after, mask)

        # 2. ΔH: エントロピー変化
        h_before = self._compute_entropy(attn_before, mask)
        h_after = self._compute_entropy(attn_after, mask)
        delta_h = h_after - h_before

        # 3. ΔSP: ショートカット純度変化
        sp_before = self._compute_shortcut_purity(attn_before, mask)
        sp_after = self._compute_shortcut_purity(attn_after, mask)
        delta_sp = sp_after - sp_before

        # F値計算
        # F = ΔEPC - λ(sign*ΔH + γΔSP)
        # 低いF = 良い変化
        #
        # entropy_sign = 1:  探索モード（ΔH>0=探索=良い→Fが下がる）
        # entropy_sign = -1: 分類モード（ΔH<0=集中=良い→-ΔHが正→Fが下がる）
        effective_delta_h = self.entropy_sign * delta_h
        F = delta_epc - self.lambda_param * (effective_delta_h + self.gamma * delta_sp)

        return GeDIGv2Result(
            F=F,
            F_mean=F.item() if F.dim() == 0 else F.mean().item(),
            delta_epc=delta_epc.item() if isinstance(delta_epc, torch.Tensor) else delta_epc,
            delta_h=delta_h.item() if isinstance(delta_h, torch.Tensor) else delta_h,
            delta_sp=delta_sp.item() if isinstance(delta_sp, torch.Tensor) else delta_sp,
            h_before=h_before.item() if isinstance(h_before, torch.Tensor) else h_before,
            h_after=h_after.item() if isinstance(h_after, torch.Tensor) else h_after,
            sp_before=sp_before.item() if isinstance(sp_before, torch.Tensor) else sp_before,
            sp_after=sp_after.item() if isinstance(sp_after, torch.Tensor) else sp_after,
        )

    def _compute_epc(
        self,
        attn_before: torch.Tensor,
        attn_after: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        有意な構造変化のみを測定

        EPCの意味:
        - 迷路: エッジの追加/削除
        - Transformer: 有意なattention変化

        閾値以下の微小変化はノイズとして無視
        """
        diff = torch.abs(attn_after - attn_before)

        if mask is not None:
            # padding部分を除外
            mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
            diff = diff * mask_2d.float()
            valid_count = mask_2d.sum().float() + 1e-9
        else:
            valid_count = float(diff.numel())

        # 閾値以上の変化をカウント
        significant = (diff > self.epc_threshold).float()
        epc = significant.sum() / valid_count

        return epc

    def _compute_entropy(
        self,
        attention: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Attention分布のエントロピー

        高H: 分散した注目（多くのトークンに薄く注目）
        低H: 集中した注目（少数のトークンに強く注目）
        """
        B, H, S, _ = attention.shape

        if mask is not None:
            # padding部分をマスク（attention計算時に使用）
            mask_2d = mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, S, 1)
            # key側のマスク
            key_mask = mask.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, S)
            attention = attention * key_mask

        # 正規化（各クエリの attention が合計1になるように）
        attn_sum = attention.sum(dim=-1, keepdim=True) + 1e-9
        attn_norm = attention / attn_sum

        # エントロピー計算
        log_attn = torch.log(attn_norm + 1e-9)
        entropy = -(attn_norm * log_attn).sum(dim=-1)  # (B, H, S)

        if mask is not None:
            # 有効なトークンのみで平均
            valid_mask = mask.unsqueeze(1).float()  # (B, 1, S)
            entropy = (entropy * valid_mask).sum() / (valid_mask.sum() * H + 1e-9)
        else:
            entropy = entropy.mean()

        return entropy

    def _compute_shortcut_purity(
        self,
        attention: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        ショートカット純度: アンカーへの経路集中度

        高い値: 少数の強い経路（効率的ショートカット）が支配
        低い値: 多数の弱い経路（ノイズ）が分散

        迷路との対応:
        - 迷路: ゴールへの経路が形成されたか
        - Transformer: CLSへの情報集約が効率化されたか
        """
        B, H, S, _ = attention.shape
        device = attention.device

        if S < 2:
            return torch.tensor(0.0, device=device)

        purities = []

        for anchor_idx in self.anchor_indices:
            # anchor_idx が負の場合の処理
            actual_idx = anchor_idx if anchor_idx >= 0 else S + anchor_idx

            if actual_idx < 0 or actual_idx >= S:
                continue

            # 内容トークン → アンカー への attention
            # アンカー自身からの attention は除外
            content_indices = [i for i in range(S) if i != actual_idx]

            if not content_indices:
                continue

            # (B, H, num_content_tokens)
            to_anchor = attention[:, :, content_indices, actual_idx]

            if mask is not None:
                # content tokens の mask を作成
                content_mask_list = [mask[:, i] for i in content_indices]
                content_mask = torch.stack(content_mask_list, dim=-1).float()  # (B, num_content)
                content_mask = content_mask.unsqueeze(1)  # (B, 1, num_content)
                to_anchor = to_anchor * content_mask

            # 上位k%の寄与率を計算
            num_content = to_anchor.shape[-1]
            k = max(1, int(num_content * self.sp_k_ratio))

            # ソートして上位kを取得
            top_k_values, _ = torch.topk(to_anchor, k, dim=-1)

            top_k_sum = top_k_values.sum(dim=-1)  # (B, H)
            total_sum = to_anchor.sum(dim=-1) + 1e-9  # (B, H)

            purity = top_k_sum / total_sum  # (B, H)
            purities.append(purity.mean())

        if not purities:
            return torch.tensor(0.0, device=device)

        return torch.stack(purities).mean()

    def compute_reference_attention(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        mask: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        参照状態（一様attention）を生成

        初期状態の近似として使用
        """
        if mask is not None:
            # マスクを考慮した一様分布
            mask_float = mask.float()  # (B, S)
            valid_len = mask_float.sum(dim=-1, keepdim=True)  # (B, 1)

            # 各位置への attention = 1 / valid_len（有効な位置のみ）
            ref_attn = mask_float.unsqueeze(1).unsqueeze(2) / (valid_len.unsqueeze(1).unsqueeze(2) + 1e-9)
            ref_attn = ref_attn.expand(batch_size, num_heads, seq_len, seq_len)
        else:
            ref_attn = torch.ones(batch_size, num_heads, seq_len, seq_len, device=device) / seq_len

        return ref_attn


class GeDIGv2Loss(nn.Module):
    """
    geDIG v2 を損失関数として使用

    Loss = CE + α * F

    F を最小化する方向に学習
    """

    def __init__(
        self,
        alpha: float = 0.1,
        **gedig_kwargs,
    ):
        super().__init__()
        self.alpha = alpha
        self.gedig = GeDIGv2(**gedig_kwargs)

    def forward(
        self,
        ce_loss: torch.Tensor,
        attn_before: torch.Tensor,
        attn_after: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, GeDIGv2Result]:
        """
        Args:
            ce_loss: Cross-entropy loss
            attn_before: 参照attention
            attn_after: 現在のattention
            mask: padding mask

        Returns:
            total_loss, gedig_result
        """
        result = self.gedig(attn_before, attn_after, mask)

        # F を損失に加算（F最小化 = 良い構造変化を促進）
        total_loss = ce_loss + self.alpha * result.F

        return total_loss, result


# =============================================================================
# テスト用ユーティリティ
# =============================================================================

def test_gedig_v2():
    """GeDIG v2 の動作確認"""
    print("=" * 60)
    print("GeDIG v2 Test")
    print("=" * 60)

    device = torch.device("cpu")
    B, H, S = 2, 4, 10  # batch, heads, seq_len

    gedig = GeDIGv2(
        lambda_param=1.0,
        gamma=0.5,
        epc_threshold=0.05,
        sp_k_ratio=0.2,
        anchor_indices=[0],  # CLS only
    )

    # テスト1: 一様分布 vs 一様分布
    print("\n[Test 1] Uniform vs Uniform")
    uniform = torch.ones(B, H, S, S) / S
    result = gedig(uniform, uniform)
    print(f"  F_mean: {result.F_mean:.4f} (expected: ~0)")
    print(f"  delta_epc: {result.delta_epc:.4f}")
    print(f"  delta_h: {result.delta_h:.4f}")
    print(f"  delta_sp: {result.delta_sp:.4f}")

    # テスト2: 一様分布 vs ショートカット形成
    # 「一部のトークンのみがCLSに強く注目」という状態を作る
    print("\n[Test 2] Uniform vs Shortcut (few tokens → CLS)")
    shortcut = torch.ones(B, H, S, S) * 0.05  # ベースは薄い
    # トークン2,3 だけがCLSに強く注目（ショートカット）
    shortcut[:, :, 2, 0] = 0.8  # token 2 → CLS
    shortcut[:, :, 3, 0] = 0.7  # token 3 → CLS
    # 各行を正規化（softmax相当）
    shortcut = shortcut / shortcut.sum(dim=-1, keepdim=True)

    result = gedig(uniform, shortcut)
    print(f"  F_mean: {result.F_mean:.4f}")
    print(f"  delta_epc: {result.delta_epc:.4f} (structure changed)")
    print(f"  delta_h: {result.delta_h:.4f}")
    print(f"  delta_sp: {result.delta_sp:.4f} (should be positive = shortcut formed)")
    print(f"  sp_before: {result.sp_before:.4f} (uniform: ~k_ratio)")
    print(f"  sp_after: {result.sp_after:.4f} (should be higher)")

    # CLS列の attention を確認
    print(f"\n  [Debug] Attention to CLS (column 0):")
    print(f"    uniform[0,0,:,0]: {uniform[0,0,:,0].tolist()}")
    print(f"    shortcut[0,0,:,0]: {[f'{x:.3f}' for x in shortcut[0,0,:,0].tolist()]}")

    # テスト3: ショートカット vs 一様分布（逆方向）
    print("\n[Test 3] Shortcut vs Uniform (reverse = losing shortcuts)")
    result = gedig(shortcut, uniform)
    print(f"  F_mean: {result.F_mean:.4f} (should be positive = bad change)")
    print(f"  delta_sp: {result.delta_sp:.4f} (should be negative = shortcut lost)")

    # テスト4: 全トークン→CLS集中 vs 一様
    # これは「ショートカット」ではなく「全経路」
    print("\n[Test 4] All tokens → CLS (not shortcut, just all paths)")
    all_to_cls = torch.zeros(B, H, S, S)
    all_to_cls[:, :, :, 0] = 0.8  # 全トークンがCLSに強く注目
    all_to_cls[:, :, :, 1:] = 0.2 / (S - 1)
    all_to_cls = all_to_cls / all_to_cls.sum(dim=-1, keepdim=True)

    result = gedig(uniform, all_to_cls)
    print(f"  F_mean: {result.F_mean:.4f}")
    print(f"  delta_sp: {result.delta_sp:.4f} (should be ~0, all tokens equal)")
    print(f"  sp_before: {result.sp_before:.4f}")
    print(f"  sp_after: {result.sp_after:.4f}")

    # テスト5: マスク付き
    print("\n[Test 5] With padding mask")
    mask = torch.ones(B, S)
    mask[:, 7:] = 0  # 最後3トークンはpadding

    result = gedig(uniform, shortcut, mask)
    print(f"  F_mean: {result.F_mean:.4f}")
    print(f"  delta_sp: {result.delta_sp:.4f}")

    # テスト6: 基本モード（延伸利得）vs アブレーション（集中利得）
    print("\n[Test 6] Baseline (extension=benefit) vs Ablation (concentration=benefit)")

    # 基本モード（entropy_sign=1）: 延伸が利得（geDIG基本原則）
    gedig_baseline = GeDIGv2(entropy_sign=1, anchor_indices=[0])  # デフォルト
    result_baseline = gedig_baseline(uniform, shortcut)
    print(f"  BASELINE (entropy_sign=1, extension=benefit):")
    print(f"    F_mean: {result_baseline.F_mean:.4f}")
    print(f"    delta_h: {result_baseline.delta_h:.4f} (negative=concentration)")
    print(f"    delta_sp: {result_baseline.delta_sp:.4f} (positive=shortcuts)")
    print(f"    → ΔH<0 reduces benefit, but ΔSP>0 adds benefit")

    # アブレーション（entropy_sign=-1）: 集中が利得
    gedig_ablation = GeDIGv2(entropy_sign=-1, anchor_indices=[0])
    result_ablation = gedig_ablation(uniform, shortcut)
    print(f"  ABLATION (entropy_sign=-1, concentration=benefit):")
    print(f"    F_mean: {result_ablation.F_mean:.4f}")
    print(f"    → ΔH<0 becomes benefit (flipped), F should be lower")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_gedig_v2()
