# CAPTCHA-Style Vision Benchmark — 150 Images

Controlled synthetic benchmark for visual binary decisions.

## Dataset
- 150 PNG grid images
- 30 each: 3x3, 3x4, 4x4, 4x5, 5x5
- Each grid: 10 easy, 10 medium, 10 hard
- Task: `Does this square contain a stoplight?`
- Ground truth is cell-level in `labels.jsonl`

## Recommended metrics
- Cell accuracy
- Precision / recall / F1
- Confidence vs. correctness
- Brier score / ECE
- Latency per decision
- Accuracy vs. grid size
- Accuracy vs. difficulty

This is a synthetic benchmark for model evaluation, not for bypassing production anti-bot systems.
