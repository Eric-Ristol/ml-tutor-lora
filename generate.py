#Interactive CLI for chatting with the fine-tuned SmolLM2 + LoRA model.
#Usage:
#   python generate.py                       -- interactive chat loop
#   python generate.py -q "What is LoRA?"   -- one question, then exit
#   python generate.py --compare             -- show base model vs fine-tuned
#The model answers machine learning questions using the style it learned during
#fine-tuning. Try asking questions that are in the dataset (it should answer
#confidently) and questions that are NOT in the dataset (to see generalization).

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import data as data_module


BASE_MODEL     = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR    = "models/lora_adapter"

MAX_NEW_TOKENS = 200    #maximum tokens the model may generate per response
TEMPERATURE    = 0.7    #controls randomness: 0 = deterministic, 1 = more varied
TOP_P          = 0.9    #nucleus sampling: only consider the top 90% probability mass


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_adapter_exists():
    #Friendly error message if the user forgot to train first.
    if not os.path.exists(ADAPTER_DIR):
        print("ERROR: No LoRA adapter found at", ADAPTER_DIR)
        print("Run  python train.py  first to fine-tune the model.")
        raise SystemExit(1)


def load_finetuned_model(device):
    #Loads the base model + LoRA adapter together.
    print("Loading model... (first run downloads ~700MB, subsequent runs are fast)")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_base_model_only(device):
    #Loads ONLY the base model without any LoRA adapter.
    #Used by --compare mode to show the difference before and after fine-tuning.
    print("Loading base model (no adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(question, tokenizer):
    #Formats the question into the SmolLM2 chat template and returns the prompt
    #string. The <|im_start|>assistant token at the end signals that the model
    #should now generate the assistant's reply.
    messages = [
        {"role": "system", "content": data_module.SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer(model, tokenizer, question, device):
    #Runs inference and returns the model's answer as a plain string.
    prompt = build_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,                     #sampling for natural-sounding output
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    #Decode only the newly generated tokens -- skip the prompt we fed in.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_loop(model, tokenizer, device):
    #Runs a simple REPL: you type a question, the model answers, repeat.
    print("\nML Tutor -- SmolLM2-360M + LoRA")
    print("Ask me anything about machine learning.")
    print("Type 'quit' or press Ctrl-C to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nModel:", end=" ", flush=True)
        answer = generate_answer(model, tokenizer, question, device)
        print(answer)
        print()


def compare_mode(device):
    #Loads both the base model and the fine-tuned model and shows their answers
    #side-by-side for a few questions. This is the clearest way to demonstrate
    #that LoRA fine-tuning actually changed the model's behaviour.
    demo_questions = [
        "What is LoRA?",
        "What is gradient descent?",
        "What is overfitting?",
    ]

    print("Loading fine-tuned model (with LoRA adapter)...")
    finetuned_model, ft_tokenizer = load_finetuned_model(device)

    print("Loading base model (no adapter)...")
    base_model, base_tokenizer = load_base_model_only(device)

    print()
    print("=" * 65)
    print("Base model  vs.  Fine-tuned model (LoRA)")
    print("=" * 65)

    for i, q in enumerate(demo_questions):
        print(f"\n[Q{i+1}] {q}\n")

        base_answer = generate_answer(base_model,      base_tokenizer, q, device)
        ft_answer   = generate_answer(finetuned_model, ft_tokenizer,   q, device)

        print("BASE MODEL :")
        print(base_answer)
        print()
        print("FINE-TUNED :")
        print(ft_answer)
        print("-" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with the fine-tuned ML Tutor (SmolLM2 + LoRA)"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question and print the answer, then exit.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show base model vs fine-tuned model side-by-side for demo questions.",
    )
    args = parser.parse_args()

    check_adapter_exists()
    device = pick_device()
    print("Device:", device)

    if args.compare:
        #Special mode: loads both models and compares their outputs.
        compare_mode(device)

    elif args.question:
        #Non-interactive: answer one question and exit.
        model, tokenizer = load_finetuned_model(device)
        answer = generate_answer(model, tokenizer, args.question, device)
        print("\nModel:", answer)

    else:
        #Default: interactive chat loop.
        model, tokenizer = load_finetuned_model(device)
        interactive_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
