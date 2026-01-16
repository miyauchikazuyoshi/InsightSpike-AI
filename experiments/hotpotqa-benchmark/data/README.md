# HotpotQA Data

## Download

```bash
python ../scripts/download_data.py
```

## Files

| File | Description | Size |
|------|-------------|------|
| `hotpotqa_distractor_dev.jsonl` | Full dev set | 7,405 examples |
| `hotpotqa_sample_100.jsonl` | Small sample for testing | 100 examples |
| `hotpotqa_sample_500.jsonl` | Medium sample for Phase 0 | 500 examples |

## Data Format

```json
{
  "id": "5a8b57f25542995d1e6f1371",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "supporting_facts": {
    "title": ["Arthur's Magazine", "First for Women"],
    "sent_id": [0, 0]
  },
  "context": {
    "title": ["Title1", "Title2", ...],
    "sentences": [["sent1", "sent2"], ["sent1", "sent2"], ...]
  },
  "type": "comparison",
  "level": "medium"
}
```

## Question Types

- **comparison**: Compare two entities
- **bridge**: Multi-hop reasoning required

## Difficulty Levels

- **easy**: Single document sufficient
- **medium**: Two documents required
- **hard**: Complex reasoning required

## Source

- [HotpotQA Official](https://hotpotqa.github.io/)
- [Paper](https://arxiv.org/abs/1809.09600)
