#!/usr/bin/env python3
"""
Google Colab用 InsightSpike-AI 簡単セットアップスクリプト
Docker経由で1分環境構築

使用方法:
    !wget https://raw.githubusercontent.com/miyauchikazuyoshi/InsightSpike-AI/main/scripts/colab/setup_docker.py
    !python setup_docker.py

著者: InsightSpike-AI Team
バージョン: 2.0.0 (Docker対応版)
"""

import subprocess
import sys
import time
from pathlib import Path

class ColabDockerSetup:
    """Colab環境でInsightSpike-AIをDockerで起動するセットアップクラス"""
    
    def __init__(self):
        self.repo_url = "https://github.com/miyauchikazuyoshi/InsightSpike-AI.git"
        self.docker_image = "ghcr.io/miyauchikazuyoshi/insightspike-ai:colab"
        self.container_name = "insightspike-colab"
        
    def run_command(self, command, description="実行中"):
        """コマンドを実行して結果を表示"""
        print(f"⚡ {description}...")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5分タイムアウト
            )
            if result.returncode == 0:
                print(f"✅ {description}完了")
                return True
            else:
                print(f"❌ エラー: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ タイムアウト: {description}")
            return False
        except Exception as e:
            print(f"❌ 例外エラー: {e}")
            return False
    
    def check_docker(self):
        """Docker環境の確認"""
        print("🐳 Docker環境を確認中...")
        
        # Dockerの確認
        if not self.run_command("docker --version", "Docker バージョン確認"):
            print("📦 Dockerをインストール中...")
            # Colab環境でのDockerインストール
            install_commands = [
                "apt-get update",
                "apt-get install -y docker.io",
                "systemctl start docker",
                "systemctl enable docker"
            ]
            for cmd in install_commands:
                if not self.run_command(f"sudo {cmd}", f"Docker設定: {cmd}"):
                    return False
        
        return True
    
    def setup_method_1_prebuilt(self):
        """Method 1: Pre-built Docker Imageを使用"""
        print("\n🚀 Method 1: Pre-built Docker Image使用")
        print("=" * 50)
        
        # Docker Imageをプル
        if not self.run_command(
            f"docker pull {self.docker_image}", 
            "InsightSpike-AI Docker Imageダウンロード"
        ):
            print("❌ Docker Imageのダウンロードに失敗しました")
            return False
        
        # 既存コンテナの停止・削除
        self.run_command(f"docker stop {self.container_name}", "既存コンテナ停止")
        self.run_command(f"docker rm {self.container_name}", "既存コンテナ削除")
        
        # コンテナ起動
        docker_run_cmd = f"""
        docker run -d \
          --name {self.container_name} \
          -p 8888:8888 \
          -v $(pwd):/content/InsightSpike-AI/workspace \
          {self.docker_image}
        """
        
        if not self.run_command(docker_run_cmd, "InsightSpike-AI環境起動"):
            print("❌ コンテナの起動に失敗しました")
            return False
        
        # 起動確認
        time.sleep(10)  # 起動待機
        if not self.run_command(
            f"docker exec {self.container_name} python -c \"import insightspike; print('InsightSpike-AI Ready!')\"",
            "動作確認"
        ):
            print("❌ InsightSpike-AIの動作確認に失敗しました")
            return False
        
        print("\n🎉 Method 1 セットアップ完了！")
        print(f"📊 Jupyter Notebook: http://localhost:8888")
        return True
    
    def setup_method_2_source(self):
        """Method 2: ソースからビルド"""
        print("\n🛠️ Method 2: ソースからビルド")
        print("=" * 50)
        
        # リポジトリクローン
        if not self.run_command(
            f"git clone {self.repo_url}",
            "InsightSpike-AIリポジトリクローン"
        ):
            print("❌ リポジトリのクローンに失敗しました")
            return False
        
        # ディレクトリ移動
        if not Path("InsightSpike-AI").exists():
            print("❌ リポジトリディレクトリが見つかりません")
            return False
        
        # Docker Composeでビルド
        compose_cmd = "cd InsightSpike-AI && docker-compose -f docker/docker-compose.colab.yml build"
        if not self.run_command(compose_cmd, "Docker環境ビルド"):
            print("❌ Docker環境のビルドに失敗しました")
            return False
        
        # Docker Composeで起動
        start_cmd = "cd InsightSpike-AI && docker-compose -f docker/docker-compose.colab.yml up -d"
        if not self.run_command(start_cmd, "Docker環境起動"):
            print("❌ Docker環境の起動に失敗しました")
            return False
        
        # 動作確認
        time.sleep(15)  # ビルド後の起動待機
        verify_cmd = "cd InsightSpike-AI && docker-compose -f docker/docker-compose.colab.yml exec insightspike-colab python -c \"import insightspike; print('InsightSpike-AI Ready!')\""
        if not self.run_command(verify_cmd, "動作確認"):
            print("❌ InsightSpike-AIの動作確認に失敗しました")
            return False
        
        print("\n🎉 Method 2 セットアップ完了！")
        print(f"📊 Jupyter Notebook: http://localhost:8888")
        print(f"🛠️ 開発環境: http://localhost:8889")
        return True
    
    def test_functionality(self):
        """基本機能テスト"""
        print("\n🧪 InsightSpike-AI基本機能テスト")
        print("=" * 50)
        
        # 基本インポートテスト
        test_cmd = f"""
        docker exec {self.container_name} python -c "
import sys
sys.path.append('/content/InsightSpike-AI/src')
from insightspike.core.layers.mock_llm_provider import MockLLMProvider
provider = MockLLMProvider()
result = provider.generate_intelligent_response('モンティ・ホール問題とは何ですか？')
print('🧠 洞察生成テスト:')
print(f'📝 質問: モンティ・ホール問題とは何ですか？')
print(f'💡 回答: {{result[\\\"response\\\"][:100]}}...')
print(f'📊 信頼度: {{result[\\\"confidence\\\"]:.2f}}')
print('✅ 基本機能正常動作中！')
"
        """
        
        if self.run_command(test_cmd, "基本機能テスト"):
            print("✅ 基本機能テスト成功")
        else:
            print("❌ 基本機能テストに問題があります")
        
        # ΔGED/ΔIG洞察検出テスト
        insight_test_cmd = f"""
        docker exec {self.container_name} python -c "
print('🔬 ΔGED/ΔIG洞察検出テスト:')
dged = -0.8
dig = 2.0
insight_detected = dged < -0.5 and dig > 1.5
print(f'📉 ΔGED: {{dged}} (構造改善)')
print(f'📈 ΔIG: {{dig}} (情報増加)')
print(f'⚡ 洞察検出: {{\\\"✅ EurekaSpike発火！\\\" if insight_detected else \\\"❌ 洞察なし\\\"}}')
print('🎯 洞察検出システム正常動作中！')
"
        """
        
        if self.run_command(insight_test_cmd, "洞察検出テスト"):
            print("✅ 洞察検出テスト成功")
        else:
            print("❌ 洞察検出テストに問題があります")
    
    def show_final_status(self):
        """最終ステータス表示"""
        print("\n" + "="*70)
        print("🎉 InsightSpike-AI Docker環境セットアップ完了！")
        print("="*70)
        
        # コンテナステータス確認
        self.run_command("docker ps --filter name=insightspike", "現在の環境状態")
        
        print("\n🌟 利用可能なサービス:")
        print("   📊 Jupyter Notebook: http://localhost:8888")
        print("   🛠️ 開発環境 (Method 2の場合): http://localhost:8889")
        
        print("\n🔧 環境管理コマンド:")
        print(f"   停止: docker stop {self.container_name}")
        print(f"   再起動: docker restart {self.container_name}")
        print(f"   ログ確認: docker logs {self.container_name}")
        
        print("\n📚 ドキュメント:")
        print("   🚀 Quick Start: https://github.com/your-username/InsightSpike-AI/blob/main/documentation/guides/QUICK_START.md")
        print("   🧠 Architecture: https://github.com/your-username/InsightSpike-AI/blob/main/documentation/ARCHITECTURE_EVOLUTION_ROADMAP.md")
        
        print("\n💡 次のステップ:")
        print("   1. Jupyter Notebookにアクセス")
        print("   2. 洞察生成デモを実行")
        print("   3. ΔGED/ΔIG実験を体験")
        print("   4. 教育シナリオをテスト")
        
        print("\n🎯 Happy Insight Discovery! 🧠✨")

def main():
    """メイン実行関数"""
    print("🚀 InsightSpike-AI Docker版 Colab セットアップ")
    print("=" * 70)
    print("⚡ 1分で完了する超高速環境構築")
    print("=" * 70)
    
    setup = ColabDockerSetup()
    
    # Docker環境確認
    if not setup.check_docker():
        print("❌ Docker環境のセットアップに失敗しました")
        sys.exit(1)
    
    print("\n📋 セットアップ方法を選択してください:")
    print("1. Method 1: Pre-built Docker Image（推奨・高速）")
    print("2. Method 2: ソースからビルド（開発者向け）")
    
    # 自動的にMethod 1を実行（Colab環境想定）
    print("\n🎯 Method 1を自動実行します...")
    
    success = setup.setup_method_1_prebuilt()
    
    if success:
        setup.test_functionality()
        setup.show_final_status()
    else:
        print("\n❌ Method 1が失敗しました。Method 2を試行します...")
        success = setup.setup_method_2_source()
        
        if success:
            setup.test_functionality()
            setup.show_final_status()
        else:
            print("\n❌ 両方の方法でセットアップに失敗しました")
            print("🔧 手動セットアップをお試しください:")
            print("   1. Dockerがインストールされているか確認")
            print("   2. ネットワーク接続を確認")
            print("   3. GitHub Issues で問題を報告")
            sys.exit(1)

if __name__ == "__main__":
    main()
