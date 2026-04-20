"""
ML Tutor – Gradio demo
Loads SmolLM2-360M-Instruct + a LoRA adapter fine-tuned on ML Q&A pairs
and serves an interactive chat interface on HuggingFace Spaces.
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── config ────────────────────────────────────────────────────────────
BASE_MODEL  = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_ID  = "EricRistol/ml-tutor-lora-adapter"

SYSTEM_PROMPT = (
    "You are a helpful AI tutor specializing in machine learning. "
    "Explain concepts clearly and concisely for students."
)

MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.7
TOP_P          = 0.9

EXAMPLE_QUESTIONS = [
    "What is gradient descent?",
    "Explain the bias-variance tradeoff.",
    "What is a learning rate?",
    "How does dropout prevent overfitting?",
    "What is the difference between L1 and L2 regularization?",
]

# ── load models once at startup ───────────────────────────────────────
device = "cpu"

print("Loading base model …")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float32, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

print("Loading LoRA adapter …")
model = PeftModel.from_pretrained(base, ADAPTER_ID)
model.eval()
print("Ready!")


# ── inference helpers ─────────────────────────────────────────────────
def build_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def answer(question: str) -> str:
    if not question.strip():
        return ""
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="ML Tutor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎓 ML Tutor\n"
        "Ask a machine-learning question and get an answer from "
        "**SmolLM2-360M** fine-tuned with **LoRA** on ML Q&A pairs.\n\n"
        "*This is a small model (360M parameters) — answers may be imperfect, "
        "but they show what LoRA fine-tuning can achieve with minimal data.*"
    )
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is gradient descent?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")
        with gr.Column(scale=4):
            output = gr.Textbox(label="Answer", lines=8, interactive=False)

    gr.Examples(
        examples=[[q] for q in EXAMPLE_QUESTIONS],
        inputs=[question],
        outputs=[output],
        fn=answer,
        cache_examples=False,
    )

    submit_btn.click(fn=answer, inputs=question, outputs=output)
    question.submit(fn=answer, inputs=question, outputs=output)

demo.launch()
