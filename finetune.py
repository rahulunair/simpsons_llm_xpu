# finetuner for simpsons dataset
import json
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="intel_extension_for_pytorch"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.io.image", lineno=13
)
from typing import List

import fire
import pandas as pd
import transformers
import torch
import intel_extension_for_pytorch as ipex
if torch.xpu.is_available():
    print("Using '{}' as an xpu device.".format(torch.xpu.get_device_name()))

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from datasets import load_dataset
from datasets import load_from_disk


# hyper params and config.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
]
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 100
OUTPUT_DIR = "experiments"
CUTOFF_LEN = 256
BASE_MODEL = "openlm-research/open_llama_3b"

model = LlamaForCausalLM.from_pretrained(BASE_MODEL)
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


def prompter(data):
  """
  Tweak and create the prompt based on data.
  """
    prompt = f"""Below is a dialogue from the TV show The Simposons and it follows this pattern.
    \n ### Instruction: Write a response the completes {data["input"]}'s line in the conversation.
    \n ### Input:
    {data["instruction"]}\n
    \n ### Response:
    {data["input"]}: {data["output"]}"""
    print(prompt)
    return prompt


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data):
    prompt = prompter(data)
    tokenized_full_prompt = tokenize(prompt)
    return tokenized_full_prompt

if __name__ == "__main__:
  config = LoraConfig(
      r=LORA_R,
      lora_alpha=LORA_ALPHA,
      target_modules=LORA_TARGET_MODULES,
      lora_dropout=LORA_DROPOUT,
      bias="none",
      task_type="CAUSAL_LM",
  )
  model = get_peft_model(model, config)
  print(print_trainable_parameters(model))
  if os.path.exists("./train_val.tkn") and os.path.exists("./val_data.tkn"):
      train_val = load_from_disk("./train_val.tkn")
      train_data = load_from_disk("./train_data.tkn")
      val_data = load_from_disk("./val_data.tkn")
  else:
      data = load_dataset("json", data_files="./isdata.json")
      train_val = data["train"].train_test_split(test_size=200, shuffle=True, seed=42)
      train_data = train_val["train"].map(generate_and_tokenize_prompt)
      val_data = train_val["test"].map(generate_and_tokenize_prompt)
      train_val.save_to_disk("./train_val.tkn")
      train_data.save_to_disk("./train_data.tkn")
      val_data.save_to_disk("./val_data.tkn")

  # Training arguments
  training_arguments = transformers.TrainingArguments(
      per_device_train_batch_size=MICRO_BATCH_SIZE,
      gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
      warmup_steps=100,
      max_steps=TRAIN_STEPS,
      learning_rate=LEARNING_RATE,
      bf16=True,                  # setting datype to bfloat16    
      logging_steps=10,
      optim="adamw_torch",
      evaluation_strategy="steps",
      save_strategy="steps",
      eval_steps=50,
      save_steps=50,
      output_dir=OUTPUT_DIR,
      save_total_limit=3,
      load_best_model_at_end=True,
      ddp_find_unused_parameters=None,
      report_to="wandb",
      no_cuda=True,               # setting cuda = False 
      use_xpu=True,               # let Trainer use available XPU device (intel GPU namespace)
      use_ipex=True,              # optimize the model and optimizer using intel extension for pyotrch (optional)
  )

  print(
      f"---\n Process rank: {training_arguments.local_rank}\n Device: {training_arguments.device}\n GPU Number: {training_arguments.n_gpu}"
      + f"\n Distributed training: {bool(training_arguments.local_rank != -1)}\n bfloat 16-bits training: {training_arguments.bf16}"
  )  
  data_collator = transformers.DataCollatorForSeq2Seq(
      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
  )

  # Huggingface Trainer config
  trainer = transformers.Trainer(
      model=model,
      train_dataset=train_data,
      eval_dataset=val_data,
      args=training_arguments,
      data_collator=data_collator,
  )
  model.config.use_cache = False
  old_state_dict = model.state_dict
  model.state_dict = (
      lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
  ).__get__(model, type(model)) 

  # train and save the model
  trainer.train()
  model.save_pretrained(OUTPUT_DIR)
