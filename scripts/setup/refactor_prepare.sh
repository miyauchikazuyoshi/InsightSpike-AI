#!/usr/bin/env bash
# プロジェクト構造リファクタリングスクリプト

set -e

echo "🔧 Starting InsightSpike-AI refactoring..."

# 1. 新しいディレクトリ構造を作成
echo "📁 Creating new directory structure..."

# Core modules
mkdir -p src/insightspike/core/{layers,agents,interfaces}
mkdir -p src/insightspike/components/{memory,graph,embedding,metrics}
mkdir -p src/insightspike/utils/{io,logging,validation}
mkdir -p src/insightspike/config

# Infrastructure
mkdir -p infrastructure/{docker,scripts,notebooks}
mkdir -p infrastructure/environments/{local,colab,production}

# Documentation
mkdir -p docs/{api,tutorials,research,deployment}

# Examples and experiments
mkdir -p examples/{basic,advanced,research}
mkdir -p experiments/{benchmarks,ablation,case_studies}

# Data and assets
mkdir -p assets/{diagrams,figures,presentations}

echo "✅ Directory structure created"

# 2. バックアップ作成
echo "💾 Creating backup..."
cp -r src/insightspike src/insightspike_backup

echo "🎉 Preparation complete!"
echo "Next: Run refactor_move_files.sh to reorganize files"
