"""
geDIG Demo - Hugging Face Spaces (Gradio version)
A unified gauge for deciding when to accept new information.
"""

import gradio as gr
import random

# Sample HotPotQA questions
SAMPLE_QUESTIONS = [
    {
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "context": [
            ["Arthur's Magazine", ["Arthur's Magazine was an American literary magazine published in the 19th century.", "It was founded in 1844."]],
            ["First for Women", ["First for Women is a women's magazine published by Bauer Media Group.", "It was launched in 1989."]]
        ],
    },
    {
        "question": "Were Pavel Urysohn and Leonid Levin known for the same type of work?",
        "answer": "Yes",
        "context": [
            ["Pavel Urysohn", ["Pavel Samuilovich Urysohn was a Soviet mathematician.", "He is best known for his contributions to topology."]],
            ["Leonid Levin", ["Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.", "He is known for his work in computational complexity theory."]]
        ],
    },
    {
        "question": "What government position was held by the woman who portrayed Edith Bunker?",
        "answer": "United States Ambassador to the United Nations",
        "context": [
            ["Jean Stapleton", ["Jean Stapleton was an American actress.", "She portrayed Edith Bunker on All in the Family."]],
            ["Shirley Temple", ["Shirley Temple Black was an American actress and diplomat.", "She served as United States Ambassador to the United Nations."]]
        ],
    },
    {
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "answer": "Paris",
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "It was constructed from 1887 to 1889."]],
            ["France", ["France is a country in Western Europe.", "Paris is the capital and most populous city of France."]]
        ],
    },
    {
        "question": "Which film has the director who was born earlier, El Dorado or Le notti bianche?",
        "answer": "Le notti bianche",
        "context": [
            ["El Dorado (1966 film)", ["El Dorado is a 1966 American Western film.", "It was directed by Howard Hawks, born in 1896."]],
            ["Le notti bianche", ["Le notti bianche is a 1957 Italian film.", "It was directed by Luchino Visconti, born in 1906."]]
        ],
    }
]

BENCHMARK_RESULTS = {
    "bm25": {"em": 0.3662, "f1": 0.5229, "latency": 820},
    "gedig": {"em": 0.3749, "f1": 0.5375, "latency": 873}
}


def simulate_gedig_evaluation(question: str) -> dict:
    """Simulate geDIG evaluation for demo purposes."""
    random.seed(hash(question) % 2**32)

    epc_norm = random.uniform(0.1, 0.4)
    delta_h = random.uniform(-0.3, 0.1)
    delta_sp = random.uniform(0.0, 0.3)
    lambda_weight = 1.0
    gamma = 1.0

    ig_norm = delta_h + gamma * delta_sp
    f_score = epc_norm - lambda_weight * ig_norm

    ag_fired = f_score > 0.2
    dg_fired = f_score < 0.1

    return {
        "f_score": f_score,
        "epc_norm": epc_norm,
        "delta_h": delta_h,
        "delta_sp": delta_sp,
        "ig_norm": ig_norm,
        "ag_fired": ag_fired,
        "dg_fired": dg_fired,
        "decision": "accept" if dg_fired else ("explore" if ag_fired else "hold")
    }


def process_question(question_idx):
    """Process selected question and return results."""
    sample = SAMPLE_QUESTIONS[question_idx]
    question = sample["question"]
    answer = sample["answer"]

    # Context display
    context_text = ""
    for title, sentences in sample["context"]:
        context_text += f"**{title}**\n"
        for sent in sentences:
            context_text += f"- {sent}\n"
        context_text += "\n"

    # geDIG evaluation
    gedig_eval = simulate_gedig_evaluation(question)

    # Results
    f_score = gedig_eval["f_score"]
    epc = gedig_eval["epc_norm"]
    ig = gedig_eval["ig_norm"]
    decision = gedig_eval["decision"]

    ag_status = "🔴 AG Fired — Need more exploration" if gedig_eval["ag_fired"] else "🟢 AG Quiet — Local structure OK"
    dg_status = "🟢 DG Fired — Shortcut confirmed!" if gedig_eval["dg_fired"] else "🟡 DG Quiet — Not yet convinced"

    # Benchmark comparison
    benchmark_md = f"""
### HotPotQA Results (7,405 questions)

| Method | EM | F1 |
|--------|-----|-----|
| BM25 | 36.6% | 52.3% |
| **geDIG** | **37.5%** | **53.8%** |
| Improvement | +0.9% | +1.5% |
"""

    # Formula breakdown
    formula_md = f"""
### Formula: F = ΔEPC - λ × (ΔH + γ × ΔSP)

| Component | Value |
|-----------|-------|
| Structural Cost (ΔEPC) | {epc:.3f} |
| Entropy Change (ΔH) | {gedig_eval['delta_h']:.3f} |
| Path Shortening (ΔSP) | {gedig_eval['delta_sp']:.3f} |
| Information Gain (IG) | {ig:.3f} |
| **F-score** | **{f_score:.3f}** |
| **Decision** | **{decision.upper()}** |
"""

    return (
        question,
        answer,
        context_text,
        f"{f_score:.3f}",
        f"{epc:.3f}",
        f"{ig:.3f}",
        decision.upper(),
        ag_status,
        dg_status,
        formula_md,
        benchmark_md
    )


# Create Gradio interface
with gr.Blocks(title="geDIG Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 geDIG Demo

    **geDIG** (Graph Edit Distance + Information Gain) is a unified gauge for deciding **when** to accept new information in RAG systems.

    ```
    F = Structural Cost − λ × Information Gain
    ```

    - **Small F** → Good update (accept)
    - **Large F** → Bad update (reject)

    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question_dropdown = gr.Dropdown(
                choices=[(f"Q{i+1}: {q['question'][:50]}...", i) for i, q in enumerate(SAMPLE_QUESTIONS)],
                value=0,
                label="Select a Question",
                interactive=True
            )

            with gr.Row():
                question_display = gr.Textbox(label="Question", interactive=False)
                answer_display = gr.Textbox(label="Ground Truth Answer", interactive=False)

            context_display = gr.Markdown(label="Available Context")

        with gr.Column(scale=1):
            gr.Markdown("### geDIG Evaluation")
            with gr.Row():
                f_score_display = gr.Textbox(label="F-score", interactive=False)
                decision_display = gr.Textbox(label="Decision", interactive=False)
            with gr.Row():
                epc_display = gr.Textbox(label="Structural Cost", interactive=False)
                ig_display = gr.Textbox(label="Information Gain", interactive=False)

            gr.Markdown("### Gate Status")
            ag_status_display = gr.Textbox(label="Ambiguity Gate (AG)", interactive=False)
            dg_status_display = gr.Textbox(label="Decision Gate (DG)", interactive=False)

    with gr.Row():
        with gr.Column():
            formula_display = gr.Markdown()
        with gr.Column():
            benchmark_display = gr.Markdown()

    gr.Markdown("""
    ---

    ### Links

    - [GitHub Repository](https://github.com/miyauchikazuyoshi/InsightSpike-AI)
    - [Paper (arXiv)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/paper/arxiv_v6_en/)
    - [5-Minute Guide](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/concepts/gedig_in_5_minutes.md)

    Contact: [@kazuyoshim5436](https://twitter.com/kazuyoshim5436)
    """)

    # Event handlers
    question_dropdown.change(
        fn=process_question,
        inputs=[question_dropdown],
        outputs=[
            question_display,
            answer_display,
            context_display,
            f_score_display,
            epc_display,
            ig_display,
            decision_display,
            ag_status_display,
            dg_status_display,
            formula_display,
            benchmark_display
        ]
    )

    # Load initial question
    demo.load(
        fn=process_question,
        inputs=[question_dropdown],
        outputs=[
            question_display,
            answer_display,
            context_display,
            f_score_display,
            epc_display,
            ig_display,
            decision_display,
            ag_status_display,
            dg_status_display,
            formula_display,
            benchmark_display
        ]
    )

if __name__ == "__main__":
    demo.launch()
