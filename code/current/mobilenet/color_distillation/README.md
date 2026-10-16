# MobileNetV2 Color Distillation

For complete architectural details, empirical layer probing benchmarks, mathematical formulations, and multi-task ONNX export specifications, see:

👉 [**Color Distillation Approach Documentation (`color_distillation_approach.md`)**](color_distillation_approach.md)

---

## Quick Reference Commands

```bash
cd vmmr/code/current/mobilenet

# 1. Run full distillation training in tmux (extracts features on first run, then trains)
bash bash_scripts/run_train_color_distillation.sh

# 2. Re-train with Linear head instead of MLP
bash bash_scripts/run_train_color_distillation.sh --head-type linear

# 3. Instant re-training on cached features (skips extraction, finishes in <2 min)
bash bash_scripts/run_train_color_distillation.sh --skip-extract --epochs 30 --lr 5e-4
```
