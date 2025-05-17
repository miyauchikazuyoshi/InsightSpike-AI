from typing import List

def build_prompt(question: str, docs: List[str]) -> str:
    context = "\n---\n".join(docs)
    return (
        "You are an expert assistant. Use the context to answer concisely.\n"
        f"[Context]\n{context}\n\n[Question]\n{question}\n\n[Answer]"
    )
