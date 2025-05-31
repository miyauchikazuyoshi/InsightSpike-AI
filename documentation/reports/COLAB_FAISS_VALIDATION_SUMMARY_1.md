# Colab Faiss-GPU Validation Strategy

## 🎯 設定要約

### 1. 依存関係管理戦略
- **CI環境**: faiss-cpu を手動pipインストール（poetry.lockとの競合回避）
- **ローカル開発**: dev groupでfaiss-cpu（poetryで管理）
- **Colab環境**: faiss-gpu を先行pipインストール、その後poetry設定

### 2. Colab Setup Workflow
```bash
# 1. faiss-gpu + sentence-transformers を最初にインストール
pip install -q faiss-gpu sentence-transformers

# 2. Poetry設定: 既存環境を使用
poetry config virtualenvs.create false

# 3. 追加依存関係をpipでインストール
pip install -q typer rich click pyyaml psutil

# 4. プロジェクトを開発モードでインストール
pip install -q -e .

# 5. Poetry環境でも利用可能にする（依存関係競合を避けるため--no-deps）
poetry install --no-deps
```

### 3. 検証スクリプト
- `scripts/colab/test_colab_env.py`: 包括的環境テスト
- PyTorch GPU, Faiss-GPU, SentenceTransformers の動作確認
- GPU acceleration パフォーマンステスト

### 4. pyproject.toml Groups
```toml
[tool.poetry.group.dev.dependencies]
faiss-cpu = "^1.7"  # 開発環境用

[tool.poetry.group.ci.dependencies]  
pytest = "^8.0"
# CIでは faiss は手動インストール

[tool.poetry.group.colab.dependencies]
faiss-gpu = "^1.7"  # Colab専用
```

## 🔍 確認ポイント

### Expected Behavior
1. ✅ faiss-gpu がColabで正常にインストールされる
2. ✅ GPU resources (`StandardGpuResources`) が利用可能
3. ✅ CPU→GPU index転送が成功する
4. ✅ GPU search操作が動作する
5. ✅ SentenceTransformersとの互換性

### Potential Issues
1. ❌ Poetry install時にfaiss-cpuがfaiss-gpuを上書き
2. ❌ 依存関係の競合でfaiss-gpuが削除される
3. ❌ GPU resourcesが利用できない（faiss-cpuがインストールされた場合）

## 🚀 テスト実行

### Colab実行コマンド
```python
# 1. リポジトリクローン
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI

# 2. セットアップ実行
!chmod +x scripts/colab/setup_colab.sh
!./scripts/colab/setup_colab.sh

# 3. 検証実行
!python scripts/colab/test_colab_env.py
```

### 成功基準
- [ ] Faiss-GPU successfully imported
- [ ] GPU resources available
- [ ] GPU index creation successful
- [ ] GPU search operations working
- [ ] Performance improvement over CPU

## 📝 次のステップ

1. **Colab実測テスト**: 実際のColab環境で動作確認
2. **パフォーマンス測定**: CPU vs GPU検索速度比較
3. **ドキュメント更新**: README にColab固有の手順追加
4. **CI/CD確認**: 現行CIワークフローの動作確認

## 🔧 万が一の対策

faiss-gpuが動作しない場合の代替案：
1. `!pip uninstall faiss-cpu -y` で明示的削除
2. `!pip install faiss-gpu --force-reinstall` で強制再インストール
3. コンテナ再起動後の再セットアップ
