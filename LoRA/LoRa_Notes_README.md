## Purpose

* **`qwen3_lora_train.py`** fine-tunes **Qwen/Qwen3-0.6B** with **LoRA** on the JSON dataset and **writes adapters + tokenizer** to `checkpoints/qwen3-0.6b-lora`.

**`qwen3_lora_infer.py`** loads the **base model + the saved LoRA adapters** and runs **text generation** on a single hard-coded prompt.

## Inputs / outputs

* **Train script inputs:** `./data/silvaco_dataset_train.json` (expects fields `instruction` and `output`). Output: adapter checkpoint directory `checkpoints/qwen3-0.6b-lora`.

* **Infer script inputs:** `BASE_MODEL_NAME` + `ADAPTER_DIR` and a hard-coded `spec` prompt string. Output: prints decoded generation to stdout.

## Tokenizer behavior

* **Training:** loads tokenizer from the *base model name* (`Qwen/Qwen3-0.6B`), sets `pad_token=eos_token` if missing, then **saves tokenizer** into `OUT_DIR`.

* **Inference:** loads tokenizer from the *adapter dir* (`ADAPTER_DIR`) specifically to match training-time tokenizer settings, and sets `pad_token` if missing.

## Model construction

* **Training:** loads base model, enables gradient checkpointing, then **wraps it with LoRA** via `LoraConfig` + `get_peft_model(...)`. Targets: `q_proj,k_proj,v_proj,o_proj`; r=16, alpha=32, dropout=0.05.

**Inference:** loads base model, then **attaches saved LoRA weights** via `PeftModel.from_pretrained(base_model, ADAPTER_DIR)` and sets `model.eval()`.

## Precision + device handling (important behavioral difference)

* **Inference script:** explicitly chooses `bfloat16` (if supported) else `float16` on CUDA else `float32`, uses `device_map="auto"` only if CUDA, and manually `.to(device)` only for CPU. It also explicitly sets `pad_token_id/eos_token_id` into `model.config` if missing.

* **Training script:** relies on `dtype="auto"` and `device_map="auto"` for the model, and separately decides whether to set `bf16=True` or `fp16=True` in `TrainingArguments` based on GPU support. It also does a compatibility hack: inspects `TrainingArguments` signature to pick `eval_strategy` vs `evaluation_strategy`.

## Data pipeline (exists only in training)

* Formats each example as: `instruction + "\n\n" + output`
* Tokenizes with `max_length=2048`, truncation, and **pads to max_length**
* Uses `DataCollatorForLanguageModeling(..., mlm=False)`
* Splits train/val with `test_size=0.2` and shuffles with seed 42

## Training loop vs generation loop

* **Training:** uses `Trainer` with cosine LR scheduler, warmup_ratio=0.05, weight_decay=0.1, grad accumulation (16) for effective batch ~64, saves each epoch, keeps 2 checkpoints.

* **Inference:** uses `model.generate(max_new_tokens=800, do_sample=False)` (greedy decoding) and prints the result.
