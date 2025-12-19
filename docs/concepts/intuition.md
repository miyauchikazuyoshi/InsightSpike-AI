# The Intuitive Guide to geDIG

> "What if your database could think like a brain?"

If you find the math in InsightSpike-AI daunting, you are in the right place. This guide explains the core concept—**geDIG**—using zero equations.

## The Analogy: The "Mental Map"

Imagine your knowledge as a **Mental Map** (a Graph) of a city.
*   **Nodes** are places (Concepts).
*   **Edges** are roads (Relationships).

Your goal is to navigate from point A to point B efficiently.

### The Scenario

You know a route from **Home** to **Work** that takes 30 minutes.
Suddenly, a friend tells you: *"Hey, if you cut through the new park, you can get there in 10 minutes!"*

This is a **Potential Update** (a "Spike" of insight).
Now, your brain has to make a decision: **Should I update my map?**

To decide, your brain weighs two things:

### 1. The Cost of Learning ($EPC$)
Updating your mental map isn't free. You have to:
*   Forget the old habit.
*   Memorize the new turn.
*   Risk getting lost if the friend is wrong.

This is **Edit-Path Cost (EPC)**. It represents the "laziness" or "inertia" of the system.
> *System says: "Ugh, changing my database structure is hard work."*

### 2. The Value of Discovery ($IG$)
If the friend is right, you save 20 minutes every single day. That's a huge gain in efficiency.

This is **Information Gain (IG)**. It represents the "curiosity" or "reward" of the new knowledge.
> *System says: "Wow! A 20-minute shortcut? That's amazing!"*

---

## The Decision Gauge ($F$)

The **geDIG** gauge is simply the balance between these two forces.

$$ F = \text{Cost} - \lambda \times \text{Value} $$

*   If **Value** is high enough to overcome the **Cost**, $F$ becomes negative (Energy drops).
*   **Decision:** **SPIKE!** (Accept the update).

### The "Personality" Slider ($\lambda$)
There is one special dial in this engine: **Lambda ($\lambda$)**. This controls the system's personality.

*   **Low $\lambda$ (The Skeptic):**
    *   Values stability over new info.
    *   "I'll stick to my old route unless the new one is *miraculously* better."
    *   Result: Robust, but slow to learn.

*   **High $\lambda$ (The Explorer):**
    *   Values discovery over stability.
    *   "A new path? Let's try it! Even if it saves just 1 minute!"
    *   Result: Fast learner, but risks "hallucinating" or getting cluttered with useless shortcuts.

## Why is this better than standard RAG?

Standard RAG (Retrieval-Augmented Generation) is like a tourist who **Googles the map every single time** they leave the house. They never learn.

**InsightSpike (geDIG)** is like a local.
1.  **Phase 1 (Awake):** It uses the map it has. If it gets confused, *then* it looks up the map (Retrieval).
2.  **Phase 2 (Sleep):** If it found a good shortcut today, it "dreams" about it and permanently draws it onto the map.

Tomorrow, it won't need to look it up. It just **knows**.
