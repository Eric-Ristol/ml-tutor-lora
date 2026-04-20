---
title: ML Tutor
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.32.0"
app_file: app.py
pinned: false
license: mit
---

# ML Tutor

An interactive ML tutor powered by **SmolLM2-360M-Instruct** fine-tuned with **LoRA** on machine learning Q&A pairs.

Ask any machine learning question and get a concise explanation.

## Model Details

- **Base model**: HuggingFaceTB/SmolLM2-360M-Instruct (360M parameters)
- **Fine-tuning**: LoRA (rank=8, alpha=16) on q_proj and v_proj
- **Training data**: 40 curated ML Q&A pairs
- **Trainable parameters**: 819,200 (0.23% of total)
