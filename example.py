# this script provides an example of how we can combine RAG, Prompt Engineering, and LoRa together to prompt the model
# and makes it easy to use the model under varying circumstances
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from rag.rag import SilvacoRAG 
from PromptEngineering.Prompts import PromptEngineer

LORA_DIR = 'LoRA/lora_adapter'
BASE_MODEL = 'Qwen/Qwen3-0.6B'
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# now we can load the base model, and the model with LoRa attached
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map="auto" if device=="cuda" else None)
base_model.eval()
print("base model loaded")

lora_tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
lora_model = PeftModel.from_pretrained(base_model, LORA_DIR)
lora_model.eval()
print('LoRa model loaded')

# now we can enable RAG, since we already have built the index we can just load it instead of rebuilding
rag = SilvacoRAG(pdf_path=None)
rag.load_index('rag/RAG')
print('rag object has been loaded')

# now we can add the prompt engineering class (no model needed)
pe = PromptEngineer(expert_role="You are an expert in Verilog HDL programming.")
print('prompt engineer loaded successfully')

# below is a simple function that allows us to test the model with different capabilities
# and should make it easy for us to see how effective our work was
# the options for prompt engineering are as follows: "few_shot", "chain_of_thought", "output_format", "decomposition"
def generate_code(user_query, use_lora = True, use_rag = True, use_prompt_engineering = None, few_shot_examples = None,
    max_new_tokens = 150,
    top_k_rag = 3,
    temperature = 0.3
    ):

    prompt = user_query
    if use_rag:
        # adjust the min score if you wish, but it often gives nothing
        rag_results = rag.retrieve(prompt, top_k=top_k_rag, min_score=0.0) 
        if rag_results:
            print(f"RAG retrieved {len(rag_results)} chunks with scores: {[r['score'] for r in rag_results]}")
            context = rag.format_context(rag_results, max_tokens=300)
            prompt = f"Reference Documentation:\n{context}\n\n---\n\nTask: {user_query}"
        else:
            print("WARNING: No RAG results found!")

    if use_prompt_engineering:

        if use_prompt_engineering == "few_shot":
            if not few_shot_examples:
                pass
            else:
                prompt = pe.few_shot(prompt, few_shot_examples, instruction="Complete the code based on the task description.")

        elif use_prompt_engineering == "chain_of_thought":
            prompt = pe.chain_of_thought(prompt, instruction="Think step-by-step about how to solve this coding task")

        elif use_prompt_engineering == "output_format":
            format_desc = "Provide only the SILVACO ATLAS commands, one per line. No explanations, no comments, no Verilog-style headers."
            prompt = pe.output_format_control(prompt, format_desc)

        elif use_prompt_engineering == "decomposition":
            prompt = pe.problem_decomposition(prompt)

    model = lora_model if use_lora else base_model
    tokenizer = lora_tokenizer if use_lora else base_tokenizer
    model_name = "LoRA" if use_lora else "Base"

    print(f"generating with {model_name} model")

    # now we can actually generate the response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # added just to extract the generated portion
    if generated_text.startswith(prompt):
        generated_code = generated_text[len(prompt):].strip()
    else:
        generated_code = generated_text

    return prompt, generated_code

if __name__ == "__main__":
    query = "Write a SILVACO ATLAS command to define a mesh with spacing of 0.1 microns"

    prompt, code = generate_code(query, use_lora=True, use_rag=True, use_prompt_engineering='output_format')

    print('PROMPT:')
    print(f"{prompt}\n")

    print("*" * 40)
    print("CODE:")
    print(f"{code}\n")