# Improved geDIG Embedding Evaluation Results

**Date**: 2025-07-03 00:38:24

**Random Seed**: 42


## Dataset Information

- Total Questions: 700
- Questions per Type: 100
- Dataset Types: squad, hotpotqa, commonsenseqa, drop, boolq, coqa, msmarco


## Performance Summary

| Method | Recall@5 | MRR | NDCG@10 | Latency (ms) |
|--------|----------|-----|---------|--------------|
| TF-IDF | 0.321 | 0.211 | 0.282 | 0.9 |
| Sentence-BERT | 0.321 | 0.211 | 0.282 | 26.7 |
| geDIG-Trainable | 0.009 | 0.012 | 0.008 | 289.7 |
| geDIG-Original | 0.277 | 0.188 | 0.244 | 1.2 |

## Statistical Significance (p-values)

Paired t-tests for Recall@5:

- TF-IDF_vs_Sentence-BERT: p=1.0000 
- TF-IDF_vs_geDIG-Trainable: p=0.0000 **
- TF-IDF_vs_geDIG-Original: p=0.0264 **
- Sentence-BERT_vs_geDIG-Trainable: p=0.0000 **
- Sentence-BERT_vs_geDIG-Original: p=0.0971 
- geDIG-Trainable_vs_geDIG-Original: p=0.0000 **

## Key Findings

1. **Best Overall Performance**: TF-IDF (Recall@5=0.321)
2. **Fastest Method**: TF-IDF (0.9ms)
3. **geDIG Improvement**: Trainable geDIG showed -96.8% improvement over original