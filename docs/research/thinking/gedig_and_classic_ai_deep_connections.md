# geDIGと初期AIの深い共通点

## 1. GPS（General Problem Solver）とgeDIGの差分思考

### GPSの中核アルゴリズム（1959年）
```python
# GPSの疑似コード
def solve_problem(current_state, goal_state):
    while current_state != goal_state:
        # 差分を計算
        difference = analyze_difference(current_state, goal_state)
        
        # 差分を減らす操作を選択
        operator = find_operator_to_reduce(difference)
        
        # 適用
        current_state = apply(operator, current_state)
```

### geDIGの中核（2024年）
```python
# geDIGの疑似コード
def generate_insight(current_graph, query):
    while not satisfactory:
        # グラフ編集距離（差分）を計算
        delta_GED = calculate_GED(previous_graph, current_graph)
        
        # エネルギーを最小化する方向を選択
        direction = minimize_energy(delta_GED, delta_IG)
        
        # グラフを更新
        current_graph = update_graph(direction)
```

### 深い共通点
1. **差分駆動型思考**
   - GPS: 「現状と目標の差」を埋める
   - geDIG: 「グラフ構造の差」を最適化

2. **段階的改善**
   - GPS: 一歩ずつ目標に近づく
   - geDIG: エネルギーを徐々に最小化

3. **状態空間の探索**
   - GPS: 問題空間での経路探索
   - geDIG: 意味空間での洞察探索

## 2. SHRDLUとgeDIGの世界モデル

### SHRDLUの内部表現（1970年）
```
WORLD-MODEL:
├─ BLOCK-A
│  ├─ color: RED
│  ├─ position: (0, 0, 0)
│  └─ on-top-of: TABLE
├─ BLOCK-B
│  ├─ color: BLUE
│  └─ on-top-of: BLOCK-A
```

### geDIGのグラフ表現（2024年）
```
KNOWLEDGE-GRAPH:
├─ Node("赤いブロック")
│  ├─ vector: [0.2, -0.5, ...]
│  └─ edges: [("上に", "青いブロック")]
├─ Node("青いブロック")
│  └─ edges: [("下に", "赤いブロック")]
```

### 深い共通点
1. **構造的知識表現**
   - SHRDLU: 関係性を明示的に保持
   - geDIG: グラフエッジで関係を表現

2. **文脈依存の理解**
   - SHRDLU: "it"が何を指すか文脈から判断
   - geDIG: 近傍ノードから文脈を構築

3. **限定世界での完全性**
   - SHRDLU: 積み木世界では完璧
   - geDIG: ベクトル空間では一貫性

## 3. Logic Theoristとgedig探索戦略

### Logic Theoristの証明探索（1956年）
```
証明木:
├─ 公理A
│  ├─ 推論規則1 → 補題X
│  │  └─ 推論規則2 → 定理Z ✓
│  └─ 推論規則3 → 行き止まり ✗
```

### geDIGの洞察探索（2024年）
```
探索空間:
├─ Query vector
│  ├─ 近傍ノード1 → 洞察候補A（低エネルギー）✓
│  │  └─ 組み合わせ → より深い洞察
│  └─ 近傍ノード2 → 矛盾（高エネルギー）✗
```

### 深い共通点
1. **ヒューリスティック探索**
   - Logic Theorist: 有望な経路を優先
   - geDIG: 低エネルギー方向を優先

2. **枝刈り**
   - Logic Theorist: 無駄な証明経路を切る
   - geDIG: 高エネルギー（矛盾）を避ける

3. **ゴール指向**
   - Logic Theorist: 定理の証明
   - geDIG: 洞察の生成

## なぜこれらの共通点が重要か

### 1. 「車輪の再発明」ではない
初期AIの本質的に正しかったアイデアを、現代技術で実現

### 2. 問題設定の普遍性
- 差分を埋める
- 構造を理解する
- 効率的に探索する

これらは知能の本質的な要素

### 3. 初期AIの限界を現代技術で克服

| 初期AIの限界 | geDIGでの解決 |
|------------|--------------|
| 離散的な記号 | 連続的なベクトル空間 |
| 手動のルール | 学習による獲得 |
| 組み合わせ爆発 | エネルギー最小化で収束 |
| 記号接地問題 | 埋め込みベクトルで接地 |

## 哲学的な含意

### 初期AI研究者の直感は正しかった
- Newell & Simon: 「物理的記号システム仮説」
- geDIG: 記号（ノード）+物理（エネルギー）

### 螺旋的発展
```
1950s: 記号AI（単純）
  ↓
1980s: 挫折
  ↓
2010s: 深層学習（複雑）
  ↓
2024: geDIG（単純な原理+複雑な表現）
```

## 結論：温故知新のAI

geDIGは初期AIの「魂」を継承している：
- **差分への着目**（GPS）
- **構造的理解**（SHRDLU）
- **探索の効率化**（Logic Theorist）

しかし、実装は全く新しい：
- ベクトル空間
- エネルギー最小化
- 自己学習

これは「先祖返り」ではなく「螺旋的進化」。
70年前の夢を、現代技術で実現している。