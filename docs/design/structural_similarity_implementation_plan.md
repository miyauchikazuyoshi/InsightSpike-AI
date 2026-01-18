# 構造的類似度評価機能 実装計画書

## 1. 目的

### 1.1 背景
現在のgeDIGは、知識グラフの「編集コスト vs 情報利得」を評価するが、
**異なるドメイン間の構造的アナロジー**を検出する機能がない。

例：「太陽系」と「原子モデル」は異なるドメインだが、
「中心＋周回体」という同じ構造パターンを持つ。
このような**構造的類似性の発見**こそが「閃き」の本質である。

### 1.2 目標
- クエリ中心のhop展開時に、サブグラフの**構造パターン**を抽出
- 異なるドメイン間で高い構造類似度を持つ場合、**アナロジーボーナス**をIGに加算
- Word2Vecの「king - queen ≈ man - woman」のグラフ構造版を実現
- 運用上は**プロトタイプ指定＋クロスドメイン前提**でアナロジーを判定（デフォルト無効）

### 1.3 期待効果
- 単なる情報検索ではなく、**創発的な洞察**を評価可能に
- cross-genre実験での「閃き」検出率向上
- geDIGの適用範囲を「整理整頓」から「発見」へ拡張

---

## 2. 設計概要

### 2.1 アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      config/models.py                        │
│  StructuralSimilarityConfig                                  │
│    - enabled, method, thresholds, weights                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              algorithms/structural_similarity.py             │
│  StructuralSimilarityEvaluator                               │
│    - signature method (hop-growth pattern)                   │
│    - spectral method (Laplacian eigenvalues)                 │
│    - WL kernel method                                        │
│    - motif method                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  algorithms/gedig_core.py                    │
│  _calculate_multihop()                                       │
│    - 各hopでStructuralSimilarityEvaluatorを呼び出し          │
│    - アナロジー検出時にIG += analogy_weight × similarity     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 手法選択

| 手法 | 概要 | 計算量 | 精度 |
|------|------|--------|------|
| **signature** | hop成長パターン + トポロジカル特徴 | O(n) | 中 |
| **spectral** | ラプラシアン固有値比較 | O(n³) | 高 |
| **wl_kernel** | Weisfeiler-Lehman カーネル | O(n·k) | 高 |
| **motif** | 3/4ノードモチーフ頻度 | O(n³) | 中 |

デフォルトは**signature**（高速かつ十分な精度）。

---

## 3. ファイル構成

### 3.1 新規作成

```
src/insightspike/
├── algorithms/
│   └── structural_similarity.py    # 構造類似度計算モジュール
│
├── config/
│   └── models.py                   # StructuralSimilarityConfig 追加
│
tests/
├── algorithms/
│   └── test_structural_similarity.py
│
├── integration/
│   └── test_gedig_structural_similarity.py
```

### 3.2 既存変更

```
src/insightspike/
├── algorithms/
│   └── gedig_core.py               # _calculate_multihop に統合
│
├── config/
│   └── models.py                   # GraphConfig に structural_similarity 追加
```

---

## 4. 詳細設計

### 4.1 StructuralSimilarityConfig

```python
class StructuralSimilarityConfig(BaseModel):
    """構造的類似度評価の設定"""

    # 有効化
    enabled: bool = Field(default=False)

    # 手法選択
    method: Literal["signature", "spectral", "wl_kernel", "motif"] = Field(
        default="signature"
    )

    # Signature method パラメータ
    max_signature_hops: int = Field(default=3, ge=1, le=5)
    include_triangles: bool = Field(default=True)
    include_clustering: bool = Field(default=True)
    include_density: bool = Field(default=True)

    # Spectral method パラメータ
    spectral_k: int = Field(default=10, ge=1, le=50)

    # WL kernel パラメータ
    wl_iterations: int = Field(default=5, ge=1, le=10)

    # アナロジー検出
    analogy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    analogy_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    cross_domain_only: bool = Field(default=True)

    # ドメイン判定用属性名
    domain_attribute: str = Field(default="domain")
```

**運用ガード**
- アナロジーボーナスは、**prototype_graph が明示されている場合のみ**付与する。
- デフォルトは `enabled=False` かつ `cross_domain_only=True` とし、実験時のみ緩和する。

### 4.2 StructuralSimilarityEvaluator

```python
@dataclass
class SimilarityResult:
    similarity: float
    method: str
    is_analogy: bool
    signature_a: Optional[np.ndarray] = None
    signature_b: Optional[np.ndarray] = None
    domain_a: Optional[str] = None
    domain_b: Optional[str] = None

class StructuralSimilarityEvaluator:
    def __init__(self, config: StructuralSimilarityConfig):
        self.config = config

    def evaluate(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        center1: Optional[str] = None,
        center2: Optional[str] = None,
    ) -> SimilarityResult:
        """2つのサブグラフの構造的類似度を評価"""
        ...

    def find_analogies(
        self,
        graph: nx.Graph,
        query_subgraph: nx.Graph,
        query_center: str,
        top_k: int = 5,
    ) -> List[SimilarityResult]:
        """グラフ内でクエリと構造的に類似したサブグラフを検索"""
        ...

    # 内部メソッド
    def _compute_signature(self, G: nx.Graph, center: Optional[str]) -> np.ndarray:
        """hop成長パターンをベクトル化"""
        ...

    def _compute_spectral(self, G: nx.Graph, k: int) -> np.ndarray:
        """ラプラシアン固有値を取得"""
        ...

    def _is_cross_domain(self, g1: nx.Graph, g2: nx.Graph) -> bool:
        """異なるドメインかどうかを判定"""
        ...
```

### 4.3 gedig_core.py への統合

```python
# GeDIGCore.__init__ に追加
self.structural_similarity_config = config.get("structural_similarity", {})
if self.structural_similarity_config.get("enabled", False):
    from .structural_similarity import StructuralSimilarityEvaluator
    self._ss_evaluator = StructuralSimilarityEvaluator(
        StructuralSimilarityConfig(**self.structural_similarity_config)
    )
else:
    self._ss_evaluator = None

# _calculate_multihop 内、combined_ig 計算後に追加
# prototype_graph / prototype_center は外部メモリやコンテキストから注入される前提
if self._ss_evaluator is not None and prototype_graph is not None:
    ss_result = self._ss_evaluator.evaluate(
        prototype_graph, sub_g2,
        center1=prototype_center,
        center2=list(focal_nodes)[0] if focal_nodes else None,
    )
    if ss_result.is_analogy:
        analogy_bonus = (
            self.structural_similarity_config.get("analogy_weight", 0.3)
            * ss_result.similarity
        )
        combined_ig += analogy_bonus
        # ログ出力
        logger.info(
            f"[ANALOGY] hop={hop} similarity={ss_result.similarity:.3f} "
            f"bonus={analogy_bonus:.3f}"
        )
```

### 4.4 実験設計の修正

実験が「構造類似の効果」を正しく測定できるように設計を補強する。

- Analogy Benchmark: テストケースごとに中心ノードを明示（hub/root など）し、必要なら複数中心の平均で安定化
- Hard Negative/Noise: 辺の追加・削除・枝刈りを入れた派生ケースを追加し、ROC/PRベースで閾値比較
- 閾値チューニング: ベースケース単位で validation/holdout を分割し、validation で閾値選択（FPR最小＋recall制約など）→ holdout で報告
- 安定性評価: 複数seedで分割を繰り返し、閾値と指標の分散を記録
- Science History Simulation: SS効果をgeDIGに反映できる形へ修正
  - 例1: source/target のSSを別計算して外部ボーナスとして合成（報告値を明示）
  - 例2: analogy_insight を中心ノードへ接続し、焦点サブグラフに構造変化が入るようにする
  - 例3: 実験内に限り cross_domain_only を false にして差分検証（本番設定とは分離）
- ドキュメント整合: README/Docstring と実際のシナリオ数を一致させる（Rutherfordを追加するか記述を削る）

---

## 5. 実装ステップ

### Phase 1: 基盤実装（MVP）

| # | タスク | 成果物 |
|---|--------|--------|
| 1-1 | StructuralSimilarityConfig 定義 | config/models.py |
| 1-2 | signature method 実装 | algorithms/structural_similarity.py |
| 1-3 | 単体テスト作成 | tests/algorithms/test_structural_similarity.py |
| 1-4 | gedig_core.py への統合 | algorithms/gedig_core.py |
| 1-5 | 統合テスト作成 | tests/integration/test_gedig_structural_similarity.py |

### Phase 1.5: 実験設計の修正

| # | タスク | 成果物 |
|---|--------|--------|
| 1.5-1 | Analogy Benchmark の中心ノード指定/複数中心平均 | experiments/structural_similarity/analogy_benchmark.py |
| 1.5-2 | Hard Negative/Noise 追加とROC/PR評価 | experiments/structural_similarity/analogy_benchmark.py |
| 1.5-3 | Science History Simulation のSS効果反映（外部ボーナス or 焦点接続） | experiments/structural_similarity/science_history_simulation.py |
| 1.5-4 | README/Docstring の整合 | experiments/structural_similarity/README.md |
| 1.5-5 | 閾値チューニング（validation/holdout split, FPR最小＋recall制約） | experiments/structural_similarity/analogy_benchmark.py |
| 1.5-6 | 安定性評価（複数seedでの閾値/指標のばらつき） | experiments/structural_similarity/analogy_benchmark.py |

### Phase 2: 手法拡張

| # | タスク | 成果物 |
|---|--------|--------|
| 2-1 | spectral method 実装 | algorithms/structural_similarity.py |
| 2-2 | motif method 実装 | algorithms/structural_similarity.py |
| 2-3 | WL kernel 実装（grakel連携） | algorithms/structural_similarity.py |
| 2-4 | 手法比較ベンチマーク | experiments/structural_similarity_benchmark/ |

### Phase 3: アナロジー検索

| # | タスク | 成果物 |
|---|--------|--------|
| 3-1 | find_analogies() 実装 | algorithms/structural_similarity.py |
| 3-2 | アナロジーキャッシュ | algorithms/analogy_cache.py |
| 3-3 | cross-genre実験での評価 | experiments/rag_cross_genre/ |

### Phase 4: 高度な機能

| # | タスク | 成果物 |
|---|--------|--------|
| 4-1 | 役割抽象化（HUB/SPOKE等） | algorithms/role_abstraction.py |
| 4-2 | 関係パターン埋め込み | algorithms/relation_embedding.py |
| 4-3 | king-queen テスト | tests/analogy/test_word2vec_style.py |

---

## 6. テスト計画

### 6.1 単体テスト

```python
# tests/algorithms/test_structural_similarity.py

def test_identical_graphs_have_similarity_1():
    """同一グラフは類似度1.0"""
    G = nx.star_graph(5)
    evaluator = StructuralSimilarityEvaluator(config)
    result = evaluator.evaluate(G, G)
    assert result.similarity == pytest.approx(1.0)

def test_star_vs_chain_low_similarity():
    """スター型とチェーン型は低類似度"""
    star = nx.star_graph(5)
    chain = nx.path_graph(6)
    result = evaluator.evaluate(star, chain)
    assert result.similarity < 0.5

def test_two_stars_high_similarity():
    """異なるサイズのスター型でも高類似度"""
    star_small = nx.star_graph(3)
    star_large = nx.star_graph(10)
    result = evaluator.evaluate(star_small, star_large)
    assert result.similarity > 0.7

def test_cross_domain_analogy_detection():
    """異ドメイン間のアナロジー検出"""
    solar = create_solar_system_graph()  # 太陽系
    atom = create_atom_graph()           # 原子モデル
    result = evaluator.evaluate(solar, atom)
    assert result.is_analogy == True
```

### 6.2 統合テスト

```python
# tests/integration/test_gedig_structural_similarity.py

def test_gedig_with_structural_similarity_enabled():
    """構造類似度有効時のgeDIG計算"""
    config = {"structural_similarity": {"enabled": True}}
    core = GeDIGCore(**config)
    # proto/center は既知構造のテンプレート（例: 太陽系）
    result = core.calculate(
        g_prev,
        g_now,
        ...,
        analogy_context={"prototype_graph": proto, "prototype_center": center},
    )
    assert "analogy_bonus" in result.metadata

def test_analogy_bonus_increases_ig():
    """アナロジー検出時にIGが増加"""
    # 構造的に類似したサブグラフを追加
    result_with_analogy = core.calculate(...)
    result_without_analogy = core_no_ss.calculate(...)
    assert result_with_analogy.ig_value > result_without_analogy.ig_value
```

### 6.3 回帰テスト

```python
def test_disabled_mode_no_change():
    """無効時は既存動作に影響なし"""
    config = {"structural_similarity": {"enabled": False}}
    core = GeDIGCore(**config)
    # 既存テストが全てパスすることを確認
```

### 6.4 実験検証

- Science History Simulation で SS 有無の geDIG/IG 差分が出ることを確認
- Analogy Benchmark の閾値スイープでROC/PRのトレードオフを記録
- 中心ノード選択の違いが結果に与える影響をログ化

---

## 7. 設定例

### 7.1 基本設定（signature method）

```yaml
# config.yaml
graph:
  structural_similarity:
    enabled: true
    method: "signature"
    max_signature_hops: 3
    include_triangles: true
    include_clustering: true
    analogy_threshold: 0.7
    analogy_weight: 0.3
    cross_domain_only: true
    domain_attribute: "category"
```

### 7.2 高精度設定（spectral method）

```yaml
graph:
  structural_similarity:
    enabled: true
    method: "spectral"
    spectral_k: 20
    analogy_threshold: 0.8
    analogy_weight: 0.4
```

### 7.3 実験用設定（全手法比較）

```yaml
graph:
  structural_similarity:
    enabled: true
    method: "signature"  # signature, spectral, wl_kernel, motif を切り替え
    analogy_threshold: 0.6  # 低めに設定して多くの候補を検出
    analogy_weight: 0.5     # 効果を強調
    cross_domain_only: false  # 同一ドメイン内も検出
```

---

## 8. 将来の拡張

### 8.1 役割抽象化

ノードを具体名から構造的役割に抽象化：

| 具体 | 抽象役割 |
|------|----------|
| 太陽, 原子核, CEO | `[HUB]` |
| 惑星, 電子, 部下 | `[SPOKE]` |
| 引力, 電磁力, 指揮系統 | `[BINDING]` |

### 8.2 関係ベクトル演算

```
king - man + woman ≈ queen
```

のグラフ版：

```
(太陽→地球) - (中心性) + (電子的性質) ≈ (原子核→電子)
```

### 8.3 アナロジー推論

発見したアナロジーを使って推論：

```
太陽系で「重力による周回」が成立
→ 原子モデルでも「電磁力による周回」が成立するはず
→ 仮説生成
```

---

## 9. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| 計算コスト増大 | レイテンシ悪化 | キャッシュ、非同期計算 |
| 誤検出（無意味なアナロジー） | ノイズ増加 | 閾値調整、cross_domain_only |
| 既存テスト破壊 | リグレッション | enabled=false がデフォルト |
| grakel 依存 | インストール複雑化 | signature をデフォルトに |

---

## 10. 成功基準

### 10.1 定量目標

- [ ] 既存テストが100%パス（リグレッションなし）
- [ ] 構造類似度計算のレイテンシ < 10ms（小規模グラフ）
- [ ] cross-genre実験でアナロジー検出率 > 30%
- [ ] Science History Simulation で SS 有無の geDIG/IG 差分が再現可能

### 10.2 定性目標

- [ ] 「太陽系 ≈ 原子モデル」のようなアナロジーを検出できる
- [ ] 設定ファイルのみで機能の有効化/手法切り替えが可能
- [ ] ドキュメントとテストが整備されている

---

## 付録: 参考文献

1. Weisfeiler-Lehman Graph Kernels (Shervashidze et al., 2011)
2. Spectral Graph Theory (Chung, 1997)
3. Network Motifs (Milo et al., 2002)
4. Graph Isomorphism Network (Xu et al., 2019)
5. Analogical Reasoning on Knowledge Graphs (various)
