import os
os.environ["IPEX_TILE_AS_DEVICE"] = "0"

import gc
import time
import warnings
import statistics

warnings.filterwarnings(
    "ignore", category=UserWarning, module="intel_extension_for_pytorch"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.io.image", lineno=13
)

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import intel_extension_for_pytorch as ipex
from fire import Fire
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

MODEL_PATH = "./model"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
CHECKPOINT_PATH = "./experiments/adapter_model.bin"


class InferenceModel:
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
        model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, config)
        checkpoint = torch.load(CHECKPOINT_PATH)
        set_peft_model_state_dict(self.model, checkpoint)
        self.model.to(DEVICE)
        self.model = ipex.optimize(model=self.model.eval(), dtype=torch.bfloat16)
        self.max_length = 100

    def generate(self, input, **kwargs):
        prompt = self.tokenizer.encode(input, add_special_tokens=False)
        inputs = torch.tensor([prompt], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            with torch.xpu.amp.autocast():
                outputs = self.model.generate(
                    input_ids=inputs,
                    do_sample=True,
                    max_length=self.max_length,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    num_beams=5,
                    repetition_penalty=1.2,
                )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def benchmark(self, num_runs=12, num_warmup=3):
        benchmark_input = "Tell me about alpacas."
        times = []
        for _ in range(num_warmup):
            self.generate(benchmark_input)
        for i in range(num_runs):
            start_time = time.time()
            self.generate(benchmark_input)
            end_time = time.time()
            if not i < 2:
                times.append(end_time - start_time)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev_time = statistics.stdev(times) if len(times) > 1 else 0
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {DEVICE}")
        print(f"Data type: FP16")
        print(f"Max tokens: {self.max_length}")
        print(f"Average time over {num_runs} runs: {avg_time} seconds")
        print(f"Min time over {num_runs} runs: {min_time} seconds")
        print(f"Max time over {num_runs} runs: {max_time} seconds")
        print(f"Standard deviation over {num_runs} runs: {std_dev_time} seconds")


def prompter(data):
    prompt = f"""Below is a dialogue from the TV show The Simposons and it follows this pattern.
    \n ### Instruction: Write a response the completes {data["input"]}'s line in the conversation.
    \n ### Input:
    {data["instruction"]}\n
    \n ### Output:"""
    return prompt


def extract_output(llm_output):
    try:
        lines = llm_output.split("\n")
        output_index = lines.index("### Output:")
        return lines[output_index + 1] if output_index + 1 < len(lines) else ""
    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        return None


def main(user_prompt=None, infer=False, bench=False):
    torch.xpu.empty_cache()
    gc.collect()
    prompts = [
        {
            "instruction": "Marge Simpson: Ooo, careful, Homer.",
            "input": "Homer Simpson",
        },
        {
            "instruction": "Marge Simpson: Ooo, careful, Homer.\nHomer Simpson: There's no time to be careful.",
            "input": "Homer Simpson",
        },
        {
            "instruction": "Marge Simpson: Sorry, Excuse us. Pardon me...",
            "input": "Homer Simpson",
        },
    ]

    model = InferenceModel()
    if infer:
        if user_prompt is not None:
            prompts = [user_prompt]
        for prompt in prompts:
            prompt = prompter(prompt)
            start_time = time.time()
            output = model.generate(prompt)
            end_time = time.time()
            print(f"{output}\n")
    if bench:
        model.benchmark()


if __name__ == "__main__":
    Fire(main)
