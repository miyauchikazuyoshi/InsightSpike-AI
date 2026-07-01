# InsightSpike-AI

## ビルド & テスト
```bash
pip install -e .          # インストール
make install-dev          # 開発依存含む
make test                 # pytest tests/
make selftest-ab          # 軽量セルフテスト
```

## コードスタイル
- フォーマッタ: black, isort
- リンター: flake8, mypy
- 変更後は `make test` で確認してからコミットする

## アーキテクチャ
- geDIG フレームワーク: 構造コストと情報利得のトレードオフを測定
- apps/ 配下にデモアプリケーション
- examples/ に使用例

## 注意事項
- モデルファイルやデータセットはコミットしない
- .env ファイルにAPIキーを保存し、gitignore済み
