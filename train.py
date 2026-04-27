#Fine-tunes SmolLM2-360M-Instruct on our ML Q&A dataset using LoRA.
#A normal fine-tune updates ALL the model weights. SmolLM2-360M has 360 million
#of them -- that requires a lot of memory and time. LoRA takes a shortcut:
#     where r is a small number like 8. So instead of d*d = 65,536 parameters
#     we only train 2*d*r = 1,024 parameters per layer. ~64x fewer!
#     The original W is frozen; only A and B change.
#     or kept separate (so you can swap adapters for different tasks).

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

import data as data_module


#The base model we start from. 360M parameters -- small enough for a laptop.
#HuggingFace will download it (~700MB) on the first run and cache it locally.
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

#Where to save the fine-tuned LoRA adapter weights after training.
ADAPTER_DIR = "models/lora_adapter"

#Kept small so the whole training run takes minutes, not hours, on a MacBook.
NUM_EPOCHS    = 3       #how many full passes over the dataset
BATCH_SIZE    = 2       #examples processed per gradient update
LEARNING_RATE = 2e-4    #step size for gradient descent
MAX_LENGTH    = 512     #maximum tokens per training example (longer gets cut off)

LORA_RANK    = 8    #r: dimension of the low-rank matrices.
                    #Higher r = more expressive but more parameters to train.
                    #r=8 is the standard starting point.
LORA_ALPHA   = 16   #scaling factor applied to the LoRA output.
LORA_DROPOUT = 0.05 #randomly zero some LoRA values during training.
                    #Small dropout is enough -- LoRA is already lightweight.


def pick_device():
    #Picks the best available compute device automatically.
    #  MPS  = Apple Silicon GPU (M1/M2/M3 Mac) -- fast and memory-efficient
    #  CUDA = Nvidia GPU (Linux/Windows gaming PC or workstation)
    #  CPU  = universal fallback -- slow but always available
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_base_model():
    #Downloads (or loads from local cache) the base model and its tokenizer.
    #The tokenizer converts text strings into integer token IDs that the model
    #understands, and back again for output.
    print("Loading tokenizer from", BASE_MODEL, "...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    #SmolLM2 does not have a dedicated padding token because it is a causal LM
    #(it was designed to generate text left-to-right, not to process batches).
    #We need one for batched training, so we reuse the end-of-sequence token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model weights...")
    #torch_dtype=float32 is the safest choice for CPU and MPS.
    #On a machine with an Nvidia GPU you could use bfloat16 to halve memory.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )

    return model, tokenizer


def apply_lora(model):
    #Wraps the base model with LoRA adapter layers using the peft library.
    #After this call, only the LoRA A and B matrices are trainable.
    #target_modules: which sub-layers get the LoRA matrices.
    #  "q_proj" = the Query projection in each attention head
    #  "v_proj" = the Value projection in each attention head
    #  These two are the standard LoRA targets. You could add more (k_proj,
    #  o_proj, gate_proj) for stronger adaptation at the cost of more parameters.
    #task_type=CAUSAL_LM: tells peft this is a text-generation model, not a
    #  classifier. It sets the correct output head and loss computation.

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",            #do not add LoRA to bias terms (not needed)
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    #This prints something like:
    #  trainable params: 786,432 || all params: 361,915,392 || trainable%: 0.217
    #Roughly 0.2% of parameters are trainable -- that is the power of LoRA.
    model.print_trainable_parameters()

    return model


def run_training():
    device = pick_device()
    print("Device:", device)
    print()

    model, tokenizer = load_base_model()

    print("\nApplying LoRA adapters...")
    model = apply_lora(model)

    #Move model to the chosen device.
    model = model.to(device)

    print("\nPreparing dataset...")
    train_dataset, test_dataset = data_module.build_hf_dataset(
        tokenizer, max_length=MAX_LENGTH
    )
    print("Train examples:", len(train_dataset))
    print("Test  examples:", len(test_dataset))

    #mlm=False means causal (next-token prediction), not masked language modelling.
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,           #ramp up the LR for the first 5% of steps
        weight_decay=0.01,           #mild L2 regularization on the LoRA weights
        logging_steps=10,            #print training loss every 10 optimizer steps
        eval_strategy="epoch",       #run evaluation after every epoch
        save_strategy="epoch",       #save a checkpoint after every epoch
        save_total_limit=2,          #keep last + best, prune older checkpoints
        load_best_model_at_end=True, #restore the best checkpoint when done
        report_to="none",            #disable wandb / tensorboard (keeps it simple)
        dataloader_pin_memory=False, #must be False for MPS compatibility
    )

    #checkpointing, and logging for us -- no manual loop needed.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
    )

    print("\n>>> Starting fine-tuning...\n")
    trainer.train()

    #To use the model later we just load the base + this adapter folder.
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("\nLoRA adapter saved to:", ADAPTER_DIR)
    print("Next steps:")
    print("  python evaluate.py   -- check perplexity + sample outputs")
    print("  python generate.py   -- interactive chat with the fine-tuned model")


if __name__ == "__main__":
    run_training()
