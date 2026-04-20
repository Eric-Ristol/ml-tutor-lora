#Evaluates the fine-tuned SmolLM2 + LoRA model.
#We measure two things:
#     Perplexity = exp(average negative log-likelihood per token).
#     The base model does not know our dataset at all, so it will be "surprised"
#     by our ML Q&A text and have HIGH perplexity. After fine-tuning the model
#     assigns higher probability to our style of answers, so perplexity DROPS.
#     A visible drop (e.g. 30 -> 8) is direct evidence that training worked.
#     We feed the fine-tuned model a few test questions and print its responses
#     next to the expected answers, so you can judge quality by eye.
#Run this AFTER python train.py has finished.

import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import data as data_module


BASE_MODEL  = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "models/lora_adapter"

#How many sample Q&A pairs to print side-by-side during evaluation.
NUM_SAMPLES = 4

#Maximum new tokens the model may generate for each sample answer.
MAX_NEW_TOKENS = 180


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_finetuned_model(device):
    #Loads the base model and then applies the saved LoRA adapter on top.
    #trained A and B matrices back into the right layers.
    if not os.path.exists(ADAPTER_DIR):
        print("ERROR: No adapter found at", ADAPTER_DIR)
        print("Run  python train.py  first.")
        raise SystemExit(1)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )

    print("Applying LoRA adapter from", ADAPTER_DIR, "...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.to(device)

    #eval() switches off dropout so the model behaves deterministically.
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, test_dataset, device):
    #Computes perplexity on the test set.
    #How the math works:
    #  - For each example we do a forward pass and get the cross-entropy loss L.
    #    L is the average of -log P(token_t | token_1..token_{t-1}) over all t.
    #  - We accumulate L * n_tokens for every example, then divide by total tokens.
    #    This gives a token-weighted average loss across the whole test set.
    #  - Perplexity = exp(average_loss).
    total_loss_x_tokens = 0.0
    total_tokens        = 0

    with torch.no_grad():   #no gradients needed during evaluation
        for example in test_dataset:
            input_ids = torch.tensor([example["input_ids"]]).to(device)
            labels    = torch.tensor([example["labels"]]).to(device)

            outputs  = model(input_ids=input_ids, labels=labels)
            n_tokens = input_ids.shape[1]

            total_loss_x_tokens += outputs.loss.item() * n_tokens
            total_tokens        += n_tokens

    average_loss = total_loss_x_tokens / total_tokens
    perplexity   = math.exp(average_loss)
    return perplexity


def generate_answer(model, tokenizer, question, device, temperature=1.0):
    #Formats a question as a chat prompt and generates the model's answer.
    messages = [
        {"role": "system", "content": data_module.SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    #add_generation_prompt=True appends the <|im_start|>assistant token so
    #the model knows it is its turn to respond.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                    #greedy decoding -- deterministic
            pad_token_id=tokenizer.eos_token_id,
        )

    #Slice off the prompt tokens so we only decode the new generated part.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_evaluation():
    device = pick_device()
    print("Device:", device)

    model, tokenizer = load_finetuned_model(device)

    #   so we get the exact same train/test split and evaluate on truly unseen data.
    print("\nPreparing test dataset...")
    _, test_dataset = data_module.build_hf_dataset(tokenizer)
    print("Test examples:", len(test_dataset))

    print("\nComputing perplexity on the test set (may take a minute)...")
    ppl = compute_perplexity(model, tokenizer, test_dataset, device)
    print()
    print("Test Perplexity:", round(ppl, 2))
    print(
        "(Lower is better. The base model typically scores 20-50 on this dataset;\n"
        " a well fine-tuned model typically drops below 5-8.)"
    )

    print()
    print("=" * 65)
    print("Sample outputs from the fine-tuned model")
    print("=" * 65)

    all_pairs = data_module.load_qa_pairs()

    #Take the last NUM_SAMPLES pairs as demo examples.
    #They are likely in the test split because we sorted by original order.
    sample_pairs = all_pairs[-NUM_SAMPLES:]

    for i, pair in enumerate(sample_pairs):
        print(f"\n[Sample {i + 1} of {NUM_SAMPLES}]")
        print("Q:", pair["question"])
        print()
        print("Expected :", pair["answer"])
        generated = generate_answer(model, tokenizer, pair["question"], device)
        print("Generated:", generated)
        print("-" * 65)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    run_evaluation()
