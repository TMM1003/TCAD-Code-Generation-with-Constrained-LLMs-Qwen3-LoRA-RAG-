import os, json, torch, inspect
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen3-0.6B"
DATA_PATH = "silvaco_dataset_train_cleaned.json"
OUT_DIR = "checkpoints/qwen3-0.6b-lora"
SEED = 42

# LoRA targets and hyperparams
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_R, LORA_A, LORA_D = 16, 32, 0.05

EPOCHS, LR, WARMUP = 3, 1e-4, 0.05
BATCH, ACCUM, MAXLEN = 4, 16, 2048  # effective batch ~64


# Tokenizer & model
tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tok.pad_token is None:
    # Qwen often lacks a pad token
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)
model.gradient_checkpointing_enable()

# LoRA config
peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_A,
    lora_dropout=LORA_D,
    target_modules=TARGETS,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)


# Data loading / preprocessing
def format_example(ex):
    return {"text": ex["instruction"].strip() + "\n\n" + ex["output"].strip()}


raw = load_dataset("json", data_files=DATA_PATH, split="train").shuffle(seed=SEED)
splits = raw.train_test_split(test_size=0.2, seed=SEED)
train_ds = splits["train"].map(format_example)
val_ds = splits["test"].map(format_example)


def tokenize(batch):
    return tok(
        batch["text"],
        max_length=MAXLEN,
        truncation=True,
        padding="max_length",
    )


train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
val_tok = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

# TrainingArguments – portable across GPU / CPU and HF versions

# Hardware / precision detection
use_cuda = torch.cuda.is_available()
bf16_supported = (
    use_cuda
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
)

# Introspect TrainingArguments signature (handles version differences)
ta_sig = inspect.signature(TrainingArguments)

train_kwargs = dict(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    seed=SEED,
)

# evaluation / eval_strategy renamed across transformers versions
if "eval_strategy" in ta_sig.parameters:
    train_kwargs["eval_strategy"] = "epoch"
elif "evaluation_strategy" in ta_sig.parameters:
    train_kwargs["evaluation_strategy"] = "epoch"

# Precision flags: only set what the machine actually supports
if "bf16" in ta_sig.parameters and bf16_supported:
    # e.g. newer GPUs with bf16
    train_kwargs["bf16"] = True
elif "fp16" in ta_sig.parameters and use_cuda:
    # e.g. RTX 2060: fp16 yes, bf16 no
    train_kwargs["fp16"] = True
# else: CPU only → stay in full precision

args = TrainingArguments(**train_kwargs)


# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tok,
    args=args,
    data_collator=collator,
    train_dataset=train_tok,
    eval_dataset=val_tok,
)

trainer.train()
trainer.save_model(OUT_DIR)  # saves adapters in this dir
tok.save_pretrained(OUT_DIR)
