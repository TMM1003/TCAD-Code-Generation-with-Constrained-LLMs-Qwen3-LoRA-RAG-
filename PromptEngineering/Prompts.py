import json
import random
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class PromptEngineer:
    def __init__(self, model_client=None, device=None, tokenizer=None, expert_role=None):
        self.model = model_client
        self.device = device
        self.tokenizer = tokenizer
        self.expert_role = expert_role or (
            "You are an expert at Silvaco TCAD code generation. "
            "You have deep knowledge of SILVACO ATLAS commands, SPICE syntax, and semiconductor device simulation. "
            "Only output valid TCAD code without explanations or comments unless specifically requested."
        )

    def few_shot(self, user_input: str, examples: list, instruction: str = "") -> str:
        prompt = f"{self.expert_role}\n\n"

        if instruction:
            prompt += f"Instruction:\n{instruction}\n\n"

        prompt += "Examples:\n"
        for ex in examples:
            prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"

        prompt += "-------------------\n"
        prompt += f"Now complete this task:\nInput: {user_input}\nOutput:"
        return self._maybe_generate(prompt)

    def chain_of_thought(
        self, user_input: str, instruction: str = "Think step-by-step about how to write the TCAD code"
    ) -> str:
        prompt = (
            f"{self.expert_role}\n\n"
            f"{instruction}. Break down the requirements, then provide the complete TCAD code solution.\n\n"
            f"Task: {user_input}\n\n"
            f"Step-by-step reasoning (max 3 steps):\n"
        )
        return self._maybe_generate(prompt)

    def output_format_control(self, user_input: str, format_description: str) -> str:
        prompt = (
            f"{self.expert_role}\n\n"
            f"Follow the required output format strictly.\n"
            f"Format: {format_description}\n\n"
            f"Task:\n{user_input}\n\n"
            f"CODE:"
        )
        return self._maybe_generate(prompt)

    def problem_decomposition(self, user_input: str) -> str:
        prompt = (
            f"{self.expert_role}\n\n"
            f"Break down this TCAD coding task into smaller subproblems.\n"
            f"For each subproblem, identify the required SILVACO commands.\n"
            f"Then provide the complete TCAD code solution.\n\n"
            f"Problem: {user_input}\n\n"
            f"Decomposition:"
        )
        return self._maybe_generate(prompt)

    def _maybe_generate(self, prompt: str) -> str:
        if self.model:
            # Tokenize prompt here
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prompt

    def loadQwen(self, MODEL_NAME="Qwen/Qwen3-0.6B"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto" if device == "cuda" else None
        )
        model.eval()
        self.model = model
        self.device = device
        self.tokenizer = tokenizer


if __name__ == "__main__":
    pe = PromptEngineer()
    pe.loadQwen()

    # Few Shot prompt example
    with open("silvaco_benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open("PromptEngineering/examples.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # examples = random.sample(full_data, 3)
    examples = full_data["tier2"]

    tier2 = benchmark["tier2_completion"]

    instructions = "Please complete the code from code_template based on the question."
    random_question = random.choice(tier2)
    user_input = (
        random_question["description"]
        + "\ncode_template:"
        + random_question["code_template"]
    )
    # print(
    #     pe.few_shot(
    #         user_input,
    #         examples,
    #         "Complete the command for the given user input. Only output the command, nothing else. Do not repeat examples",
    #     )
    # )

    # ---------------------------------------------------------

    # Chain of thought
    # print(pe.chain_of_thought(user_input, instructions))

    # ---------------------------------------------------------

    # Output_format_control
    format = (
        "Fill in the blanks with the correct elements according to the task description:\n"
        "- Use the description to determine what each blank represents\n"
        "- Follow the rules in the scoring_criteria (e.g., node order , parameter syntax)\n"
        "- Keep proper units where relevant\n"
        "Return the result exactly as a single line matching the template"
    )
    # print(pe.output_format_control(user_input, format))

    # ----------------------------------------------------------

    # Problem Decomposition
    print(pe.problem_decomposition(user_input))

    print("Expected Answers:")
    print(random_question["expected_answer"])
