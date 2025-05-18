import sys
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from insightspike.layer2_memory_manager import Memory
from insightspike.agent_loop import cycle
from insightspike.loader import load_corpus

def get_question():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input("質問文を入力してください: ")

def ensure_memory():
    try:
        return Memory.load()
    except FileNotFoundError:
        docs = load_corpus()
        mem = Memory.build(docs)
        mem.save()
        return mem

def main():
    try:
        # 1. 質問文投入
        question = get_question()

        # 2. エピソード記憶ロード（なければ作成）
        mem = ensure_memory()

        # 3. L1-L4ループ実行
        loop_num = 10
        eureka_spikes, ged_list, ig_list, updated_episodes = [], [], [], []
        g_old = None

        for i in range(loop_num):
            print(f"\n=== Loop {i+1} ===")
            result = cycle(mem, question, g_old)
            g_new = result.get("graph", None)
            ged = result.get("delta_ged", None)
            ig = result.get("delta_ig", None)
            eureka = result.get("eureka", False)
            updated = result.get("updated_episodes", [])

            ged_list.append(ged)
            ig_list.append(ig)
            eureka_spikes.append(eureka)
            updated_episodes.append(updated)
            g_old = g_new

            print(f"ΔGED: {ged}, ΔIG: {ig}, Eureka: {eureka}")
            print(f"更新エピソード数: {len(updated)}")

        # 4. ビジュアライズ
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(ged_list, label="ΔGED")
        plt.plot(ig_list, label="ΔIG")
        plt.xlabel("Loop")
        plt.ylabel("Value")
        plt.legend()
        plt.title("グラフ更新量の推移")

        plt.subplot(1, 2, 2)
        plt.bar(range(loop_num), [int(e) for e in eureka_spikes])
        plt.xlabel("Loop")
        plt.ylabel("Eureka Spike")
        plt.title("エウレカスパイク発生タイミング")
        plt.tight_layout()
        plt.show()

        # 更新エピソードのサマリ表示
        print("\n=== 更新されたエピソード一覧 ===")
        for i, episodes in enumerate(updated_episodes):
            print(f"\n[Loop {i+1}]")
            for ep in episodes:
                print(f"- テキスト: {ep.get('text', '')[:40]}... | 変更タイプ: {ep.get('change_type', '')} | C値: {ep.get('c_value', '')}")

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e.filename}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()

