"""
新規アナロジー発見実験

目的: 人間がまだ気づいていない（かもしれない）構造的類似性を発見する

方法:
1. 多様なドメインから知識グラフを構築
2. 全ペア間で同型発見アルゴリズムを実行
3. 低コストで、かつ「意外な」ペアを抽出
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any
from itertools import combinations
import networkx as nx

from src.insightspike.algorithms.isomorphism_discovery import discover_insight, Transform


@dataclass
class KnowledgeGraph:
    """ドメイン固有の知識グラフ"""
    name: str
    domain: str
    subdomain: str
    graph: nx.Graph
    description: str


@dataclass
class AnalogyCandidate:
    """発見されたアナロジー候補"""
    source: str
    target: str
    source_domain: str
    target_domain: str
    transform_cost: float
    node_mapping: Dict[str, str]
    insight_description: str
    novelty_score: float  # 高いほど意外


def create_diverse_knowledge_base() -> List[KnowledgeGraph]:
    """多様なドメインから知識グラフを構築"""
    graphs = []

    # =====================================================
    # 物理学
    # =====================================================

    # 波動
    wave = nx.Graph()
    wave.add_node("source", type="generator")
    wave.add_node("medium", type="transmitter")
    wave.add_node("amplitude", type="property")
    wave.add_node("frequency", type="property")
    wave.add_node("wavelength", type="property")
    wave.add_edge("source", "medium")
    wave.add_edge("medium", "amplitude")
    wave.add_edge("medium", "frequency")
    wave.add_edge("frequency", "wavelength")
    graphs.append(KnowledgeGraph(
        "wave", "physics", "mechanics",
        wave, "波動現象の基本構造"
    ))

    # 場
    field = nx.Graph()
    field.add_node("source", type="generator")
    field.add_node("space", type="transmitter")
    field.add_node("strength", type="property")
    field.add_node("gradient", type="property")
    field.add_node("potential", type="property")
    field.add_edge("source", "space")
    field.add_edge("space", "strength")
    field.add_edge("space", "gradient")
    field.add_edge("gradient", "potential")
    graphs.append(KnowledgeGraph(
        "field", "physics", "field_theory",
        field, "場の基本構造"
    ))

    # =====================================================
    # 生物学
    # =====================================================

    # 免疫系
    immune = nx.DiGraph()
    immune.add_node("antigen", role="threat")
    immune.add_node("detector", role="sensor")
    immune.add_node("signal", role="messenger")
    immune.add_node("effector", role="responder")
    immune.add_node("memory", role="storage")
    immune.add_edge("antigen", "detector")
    immune.add_edge("detector", "signal")
    immune.add_edge("signal", "effector")
    immune.add_edge("effector", "memory")
    graphs.append(KnowledgeGraph(
        "immune_system", "biology", "immunology",
        immune, "免疫応答の基本構造"
    ))

    # 遺伝子発現
    gene_expr = nx.DiGraph()
    gene_expr.add_node("stimulus", role="trigger")
    gene_expr.add_node("receptor", role="sensor")
    gene_expr.add_node("cascade", role="amplifier")
    gene_expr.add_node("transcription", role="executor")
    gene_expr.add_node("protein", role="product")
    gene_expr.add_edge("stimulus", "receptor")
    gene_expr.add_edge("receptor", "cascade")
    gene_expr.add_edge("cascade", "transcription")
    gene_expr.add_edge("transcription", "protein")
    graphs.append(KnowledgeGraph(
        "gene_expression", "biology", "molecular",
        gene_expr, "遺伝子発現の基本構造"
    ))

    # =====================================================
    # 情報科学
    # =====================================================

    # コンパイラ
    compiler = nx.DiGraph()
    compiler.add_node("source_code", role="input")
    compiler.add_node("lexer", role="parser")
    compiler.add_node("ast", role="intermediate")
    compiler.add_node("optimizer", role="transformer")
    compiler.add_node("machine_code", role="output")
    compiler.add_edge("source_code", "lexer")
    compiler.add_edge("lexer", "ast")
    compiler.add_edge("ast", "optimizer")
    compiler.add_edge("optimizer", "machine_code")
    graphs.append(KnowledgeGraph(
        "compiler", "cs", "systems",
        compiler, "コンパイラの基本構造"
    ))

    # 機械学習パイプライン
    ml_pipeline = nx.DiGraph()
    ml_pipeline.add_node("raw_data", role="input")
    ml_pipeline.add_node("preprocessor", role="parser")
    ml_pipeline.add_node("features", role="intermediate")
    ml_pipeline.add_node("model", role="transformer")
    ml_pipeline.add_node("prediction", role="output")
    ml_pipeline.add_edge("raw_data", "preprocessor")
    ml_pipeline.add_edge("preprocessor", "features")
    ml_pipeline.add_edge("features", "model")
    ml_pipeline.add_edge("model", "prediction")
    graphs.append(KnowledgeGraph(
        "ml_pipeline", "cs", "ml",
        ml_pipeline, "機械学習パイプラインの基本構造"
    ))

    # =====================================================
    # 経済学
    # =====================================================

    # 市場
    market = nx.Graph()
    market.add_node("supply", role="source")
    market.add_node("demand", role="sink")
    market.add_node("price", role="mediator")
    market.add_node("quantity", role="flow")
    market.add_node("equilibrium", role="state")
    market.add_edge("supply", "price")
    market.add_edge("demand", "price")
    market.add_edge("price", "quantity")
    market.add_edge("quantity", "equilibrium")
    graphs.append(KnowledgeGraph(
        "market", "economics", "microeconomics",
        market, "市場メカニズムの基本構造"
    ))

    # 金融システム
    finance = nx.DiGraph()
    finance.add_node("depositor", role="source")
    finance.add_node("bank", role="intermediary")
    finance.add_node("borrower", role="sink")
    finance.add_node("interest", role="flow")
    finance.add_node("risk", role="constraint")
    finance.add_edge("depositor", "bank")
    finance.add_edge("bank", "borrower")
    finance.add_edge("bank", "interest")
    finance.add_edge("borrower", "risk")
    graphs.append(KnowledgeGraph(
        "banking", "economics", "finance",
        finance, "銀行システムの基本構造"
    ))

    # =====================================================
    # 社会科学
    # =====================================================

    # 情報拡散
    info_spread = nx.DiGraph()
    info_spread.add_node("origin", role="source")
    info_spread.add_node("early_adopter", role="amplifier")
    info_spread.add_node("majority", role="mass")
    info_spread.add_node("laggard", role="tail")
    info_spread.add_node("saturation", role="limit")
    info_spread.add_edge("origin", "early_adopter")
    info_spread.add_edge("early_adopter", "majority")
    info_spread.add_edge("majority", "laggard")
    info_spread.add_edge("laggard", "saturation")
    graphs.append(KnowledgeGraph(
        "information_diffusion", "sociology", "communication",
        info_spread, "情報拡散の基本構造"
    ))

    # 革命
    revolution = nx.DiGraph()
    revolution.add_node("grievance", role="trigger")
    revolution.add_node("vanguard", role="catalyst")
    revolution.add_node("mobilization", role="amplifier")
    revolution.add_node("confrontation", role="crisis")
    revolution.add_node("transformation", role="outcome")
    revolution.add_edge("grievance", "vanguard")
    revolution.add_edge("vanguard", "mobilization")
    revolution.add_edge("mobilization", "confrontation")
    revolution.add_edge("confrontation", "transformation")
    graphs.append(KnowledgeGraph(
        "revolution", "sociology", "political",
        revolution, "革命の基本構造"
    ))

    # =====================================================
    # 化学
    # =====================================================

    # 触媒反応
    catalysis = nx.DiGraph()
    catalysis.add_node("reactant", role="input")
    catalysis.add_node("catalyst", role="enabler")
    catalysis.add_node("intermediate", role="transition")
    catalysis.add_node("product", role="output")
    catalysis.add_node("energy_barrier", role="constraint")
    catalysis.add_edge("reactant", "catalyst")
    catalysis.add_edge("catalyst", "intermediate")
    catalysis.add_edge("intermediate", "product")
    catalysis.add_edge("catalyst", "energy_barrier")
    graphs.append(KnowledgeGraph(
        "catalysis", "chemistry", "kinetics",
        catalysis, "触媒反応の基本構造"
    ))

    # =====================================================
    # 心理学
    # =====================================================

    # 学習
    learning = nx.DiGraph()
    learning.add_node("stimulus", role="input")
    learning.add_node("attention", role="filter")
    learning.add_node("encoding", role="processing")
    learning.add_node("consolidation", role="storage")
    learning.add_node("retrieval", role="output")
    learning.add_edge("stimulus", "attention")
    learning.add_edge("attention", "encoding")
    learning.add_edge("encoding", "consolidation")
    learning.add_edge("consolidation", "retrieval")
    graphs.append(KnowledgeGraph(
        "learning", "psychology", "cognitive",
        learning, "学習プロセスの基本構造"
    ))

    # 感情
    emotion = nx.DiGraph()
    emotion.add_node("event", role="trigger")
    emotion.add_node("appraisal", role="evaluation")
    emotion.add_node("arousal", role="physiological")
    emotion.add_node("feeling", role="subjective")
    emotion.add_node("behavior", role="response")
    emotion.add_edge("event", "appraisal")
    emotion.add_edge("appraisal", "arousal")
    emotion.add_edge("arousal", "feeling")
    emotion.add_edge("feeling", "behavior")
    graphs.append(KnowledgeGraph(
        "emotion", "psychology", "affective",
        emotion, "感情プロセスの基本構造"
    ))

    # =====================================================
    # 芸術・音楽
    # =====================================================

    # 物語構造
    narrative = nx.DiGraph()
    narrative.add_node("exposition", role="setup")
    narrative.add_node("rising_action", role="development")
    narrative.add_node("climax", role="peak")
    narrative.add_node("falling_action", role="resolution")
    narrative.add_node("denouement", role="conclusion")
    narrative.add_edge("exposition", "rising_action")
    narrative.add_edge("rising_action", "climax")
    narrative.add_edge("climax", "falling_action")
    narrative.add_edge("falling_action", "denouement")
    graphs.append(KnowledgeGraph(
        "narrative", "arts", "literature",
        narrative, "物語の基本構造（フライタークのピラミッド）"
    ))

    # ソナタ形式
    sonata = nx.DiGraph()
    sonata.add_node("exposition", role="presentation")
    sonata.add_node("development", role="elaboration")
    sonata.add_node("recapitulation", role="return")
    sonata.add_node("coda", role="conclusion")
    sonata.add_edge("exposition", "development")
    sonata.add_edge("development", "recapitulation")
    sonata.add_edge("recapitulation", "coda")
    graphs.append(KnowledgeGraph(
        "sonata", "arts", "music",
        sonata, "ソナタ形式の基本構造"
    ))

    # =====================================================
    # 地質学
    # =====================================================

    # 岩石サイクル
    rock_cycle = nx.DiGraph()
    rock_cycle.add_node("magma", role="source")
    rock_cycle.add_node("igneous", role="form1")
    rock_cycle.add_node("sediment", role="intermediate")
    rock_cycle.add_node("sedimentary", role="form2")
    rock_cycle.add_node("metamorphic", role="form3")
    rock_cycle.add_edge("magma", "igneous")
    rock_cycle.add_edge("igneous", "sediment")
    rock_cycle.add_edge("sediment", "sedimentary")
    rock_cycle.add_edge("sedimentary", "metamorphic")
    rock_cycle.add_edge("metamorphic", "magma")
    graphs.append(KnowledgeGraph(
        "rock_cycle", "geology", "petrology",
        rock_cycle, "岩石サイクルの基本構造"
    ))

    return graphs


def calculate_novelty_score(source: KnowledgeGraph, target: KnowledgeGraph) -> float:
    """
    新規性スコアを計算

    高いスコア = より意外なアナロジー
    """
    # 異なるドメイン間の方が新規性が高い
    if source.domain == target.domain:
        domain_bonus = 0.0
    else:
        domain_bonus = 0.5

    # サブドメインも異なるとさらに高い
    if source.subdomain != target.subdomain:
        subdomain_bonus = 0.3
    else:
        subdomain_bonus = 0.0

    # 既知のアナロジーペアを低くスコア
    known_pairs = [
        ("wave", "field"),  # 物理学内は既知
        ("compiler", "ml_pipeline"),  # CS内は既知
        ("immune_system", "gene_expression"),  # 生物学内は既知
    ]
    if (source.name, target.name) in known_pairs or (target.name, source.name) in known_pairs:
        known_penalty = -0.3
    else:
        known_penalty = 0.0

    # 「古典的」なアナロジーも低くスコア
    classic_domains = [
        ("physics", "biology"),  # 古典的
        ("biology", "economics"),  # 生態系-経済は古典的
    ]
    if (source.domain, target.domain) in classic_domains or (target.domain, source.domain) in classic_domains:
        classic_penalty = -0.1
    else:
        classic_penalty = 0.0

    # ランダム要素（同点回避）
    import random
    random.seed(hash(source.name + target.name))
    random_factor = random.random() * 0.1

    return domain_bonus + subdomain_bonus + known_penalty + classic_penalty + random_factor


def discover_novel_analogies(
    knowledge_base: List[KnowledgeGraph],
    max_cost: float = 2.0,
    top_k: int = 10
) -> List[AnalogyCandidate]:
    """
    新規アナロジーを発見

    Args:
        knowledge_base: 知識グラフのリスト
        max_cost: この値以下のコストのペアのみ考慮
        top_k: 上位k個を返す
    """
    candidates = []

    print(f"Comparing {len(knowledge_base)} graphs ({len(list(combinations(knowledge_base, 2)))} pairs)...")

    for kg1, kg2 in combinations(knowledge_base, 2):
        # 同じドメイン内のペアはスキップしない（意外な発見もあり得る）

        transform = discover_insight(kg1.graph, kg2.graph)

        if transform.cost <= max_cost:
            novelty = calculate_novelty_score(kg1, kg2)

            candidate = AnalogyCandidate(
                source=kg1.name,
                target=kg2.name,
                source_domain=f"{kg1.domain}/{kg1.subdomain}",
                target_domain=f"{kg2.domain}/{kg2.subdomain}",
                transform_cost=transform.cost,
                node_mapping=transform.node_mapping,
                insight_description=transform.to_insight_description(),
                novelty_score=novelty
            )
            candidates.append(candidate)

    # 新規性スコアでソート（高い順）、同点ならコストで（低い順）
    candidates.sort(key=lambda x: (-x.novelty_score, x.transform_cost))

    return candidates[:top_k]


def format_discovery_report(candidates: List[AnalogyCandidate]) -> str:
    """発見レポートをフォーマット"""
    lines = []
    lines.append("=" * 70)
    lines.append("新規アナロジー発見レポート")
    lines.append("=" * 70)
    lines.append("")

    for i, c in enumerate(candidates, 1):
        lines.append(f"### 発見 #{i}: {c.source} ≈ {c.target}")
        lines.append(f"    ドメイン: {c.source_domain} ↔ {c.target_domain}")
        lines.append(f"    変換コスト: {c.transform_cost}")
        lines.append(f"    新規性スコア: {c.novelty_score:.3f}")
        lines.append(f"    マッピング:")
        for src, tgt in c.node_mapping.items():
            lines.append(f"      {src} → {tgt}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("新規アナロジー発見実験を開始...")
    print()

    # 1. 知識ベースを構築
    knowledge_base = create_diverse_knowledge_base()
    print(f"知識ベース: {len(knowledge_base)} グラフ")
    for kg in knowledge_base:
        print(f"  - {kg.name} ({kg.domain}/{kg.subdomain}): {len(kg.graph.nodes())} nodes")
    print()

    # 2. 新規アナロジーを発見
    candidates = discover_novel_analogies(knowledge_base, max_cost=1.0, top_k=15)

    # 3. レポート出力
    report = format_discovery_report(candidates)
    print(report)

    # 4. 結果を保存
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    results = {
        "knowledge_base_size": len(knowledge_base),
        "candidates_found": len(candidates),
        "candidates": [asdict(c) for c in candidates]
    }
    with open(output_dir / "novel_analogies.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存: {output_dir / 'novel_analogies.json'}")

    # 5. ハイライト
    print("\n" + "=" * 70)
    print("最も新規性の高い発見 TOP 3")
    print("=" * 70)
    for i, c in enumerate(candidates[:3], 1):
        print(f"\n{i}. {c.source} ≈ {c.target}")
        print(f"   「{c.source}」と「{c.target}」は構造的に同型！")
        print(f"   これは {c.source_domain} と {c.target_domain} の間の")
        print(f"   意外なアナロジーかもしれません。")


if __name__ == "__main__":
    main()
