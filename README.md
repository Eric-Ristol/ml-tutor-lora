# ML Tutor with LoRA

Fine-tune SmolLM2-360M-Instruct on ML education Q&A using LoRA.

**[Live demo](https://huggingface.co/spaces/EricRistol/ml-tutor)**

## What is LoRA

Normal fine-tuning updates all 360 million model weights. Takes huge memory and hours on GPU.

LoRA (Low-Rank Adaptation) keeps the original weights frozen and injects two tiny trainable matrices (A and B) into each attention layer. Only A and B get updated during training.

With rank r=8:
- Normal attention weight: d×d = 65,536 parameters
- With LoRA: 2×d×r = 1,024 trainable parameters
- That's 64x fewer parameters to train

The adapter file ends up being a few MB instead of 700 MB.

## What's included

- `data.py`: 40 ML Q&A pairs baked in (no downloads needed)
- `train.py`: Fine-tune with LoRA for 3 epochs
- `evaluate.py`: Compute perplexity and sample outputs
- `generate.py`: Chat with the fine-tuned model
- `--compare` flag: Compare base model vs fine-tuned side-by-side

## How to run

```bash
pip install -r requirements.txt
```

First run downloads SmolLM2 (~700 MB, cached after that).

### Inspect the dataset

```bash
python data.py
```

Saves 40 Q&A pairs to `data/qa_dataset.json`.

### Fine-tune

```bash
python train.py
```

Trains for 3 epochs. Expected time: 5-15 minutes on CPU, 2-5 minutes on Apple Silicon.

### Evaluate

```bash
python evaluate.py
```

Computes perplexity on test examples and prints sample generations.

### Chat

```bash
python generate.py
```

Interactive chat with the fine-tuned model.

### Compare models

```bash
python generate.py --compare
```

Shows base model vs fine-tuned model side-by-side.

## Results

Fine-tuned model achieves lower perplexity and more focused, correct answers compared to the base model.

## Files

```
├── data.py              Q&A dataset
├── train.py             LoRA fine-tuning
├── evaluate.py          perplexity + sampling
├── generate.py          interactive chat
├── main.py              CLI menu
├── hf-space/            Gradio demo for HuggingFace
├── models/lora_adapter/ saved LoRA weights
└── data/qa_dataset.json the training data
```

---

**[Live demo](https://huggingface.co/spaces/EricRistol/ml-tutor)**
