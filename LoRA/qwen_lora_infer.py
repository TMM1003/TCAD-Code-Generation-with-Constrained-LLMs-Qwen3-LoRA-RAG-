# qwen3_lora_infer.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = "checkpoints/qwen3-0.6b-lora"

# Device / precision setup

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

# bf16 is only on newer GPUs; for portability, fall back to fp16/fp32
bf16_supported = (
    use_cuda
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
)

if bf16_supported:
    load_dtype = torch.bfloat16
elif use_cuda:
    load_dtype = torch.float16
else:
    load_dtype = torch.float32


# Tokenizer

# Load tokenizer from adapter dir so you get the same vocab/settings as training
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# Base model + LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=load_dtype,
    device_map="auto" if use_cuda else None,  # let HF place on GPU if available
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# If we're on pure CPU and didn't use device_map, move to device explicitly
if not use_cuda:
    model.to(device)

# Make sure model has pad/eos ids set consistently
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id
if model.config.eos_token_id is None:
    model.config.eos_token_id = tok.eos_token_id

# Inference
# Change spec to whatever prompt you want to test
spec = (
    "Generate a Silvaco/SmartSpice deck for an NMOS L=180nm W=1um, "
    "VDD=1.8V. Include mesh, models, contacts, bias sources, and a DC sweep "
    "of VGS from 0 to 1.8V."
)

inputs = tok(spec, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=800,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

print(tok.decode(output_ids[0], skip_special_tokens=True))