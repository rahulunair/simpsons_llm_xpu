# finetuner for simpsons dataset
import os
import warnings

os.environ["IPEX_TILE_AS_DEVICE"] = "0"

warnings.filterwarnings(
    "ignore", category=UserWarning, module="intel_extension_for_pytorch"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.io.image", lineno=13
)

import json
import sys
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import intel_extension_for_pytorch as ipex
import fire
import pandas as pd
import wandb

from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model


if torch.xpu.is_available():
    print("Using '{}' as an xpu device.".format(torch.xpu.get_device_name()))


# hyper params and config.
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 100
OUTPUT_DIR = "experiments"
CUTOFF_LEN = 256
# BASE_MODEL = "openlm-research/open_llama_3b"
# BASE_MODEL = "Writer/camel-5b-hf"
BASE_MODEL = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
EPOCHS = 10  # Please set the number of epochs you want to train for
device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
generate_while_training = False
N = 64
dtype = torch.bfloat16

# Initialize wandb with project name and config parameters
wandb.init(
    project="simpsons_llm_finetuning_no_trainer",
    config={
        "model_name": BASE_MODEL,
        "data_type": "json",
        "device_type": str(device),
        "batch_size": str(BATCH_SIZE),
        "micro_batch_size": str(MICRO_BATCH_SIZE),
    },
)

print(f"Using model: {BASE_MODEL}")
if BASE_MODEL.startswith("openlm"):
    model = LlamaForCausalLM.from_pretrained(BASE_MODEL)
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    print("using llama tokenizer and model class...")
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL
    )  # , torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
wandb.watch(model, log="all")

config = LoraConfig(
    r=64,
    lora_alpha=32,
    # target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)


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
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def generate_text(model, prompt, max_length=100):
    # Encoding the input prompt
    encoded_prompt = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    model = model.to(device).to(torch.float32)
    # Generate text
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
    )

    # Decoding the output
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    return text


def infer_prompt():
    """
    infer prompt to test the quality of the model
    """
    prompt = f"""Below is a dialogue from the TV show The Simposons and it follows this pattern.
    \n ### Instruction: Write a response the completes Homer's line in the conversation.
    \n ### Input:
    " Homer Simpson: D'oh!"\n
    \n ### Response: """
    return prompt


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


def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    # Pad sequences
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if os.path.exists("./train_val.tkn") and os.path.exists("./val_data.tkn"):
    train_val = load_from_disk("./train_val.tkn")
    train_data = load_from_disk("./train_data.tkn")
    val_data = load_from_disk("./val_data.tkn")
else:
    data = load_dataset("json", data_files="./dataset/isdata.json")
    train_val = data["train"].train_test_split(test_size=200, shuffle=True, seed=42)
    train_data = train_val["train"].map(generate_and_tokenize_prompt)
    val_data = train_val["test"].map(generate_and_tokenize_prompt)
    train_val.save_to_disk("./train_val.tkn")
    train_data.save_to_disk("./train_data.tkn")
    val_data.save_to_disk("./val_data.tkn")

# Initialize the DataLoader for training and validation datasets
train_dataloader = DataLoader(
    train_data, batch_size=MICRO_BATCH_SIZE, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_data, batch_size=MICRO_BATCH_SIZE, collate_fn=collate_fn
)

# Choose a loss function
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id
)  # Ignore padding token for loss computation

# Choose an optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Set up a linear learning rate scheduler
num_training_steps = len(train_dataloader) * EPOCHS  # Total number of training steps
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)

model = model.to(device)

# Generate text before training
model.eval()
prompt = infer_prompt()
generated_text = generate_text(model, prompt)
print(f"Generated text before training:\n{generated_text}")
model.train()
if torch.xpu.is_available():
    torch.xpu.empty_cache()
    print("XPU cache released")

optimize = False
# Training loop
model = model.to(memory_format=torch.channels_last)

print("How many params are trainable before adding Lora...")
print_trainable_parameters(model)
model = get_peft_model(model, config)
print("How many params are trainable after adding Lora...")
print_trainable_parameters(model)

# move peft updates to model state dict
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 10)

    # Training
    model.train()
    if not optimize:
        optimize = True
        if hasattr(ipex, "optimize_transformers"):
            # Use the llm_optimize function
            model, optimizer = ipex.optimize_transformers(
                model=model, optimizer=optimizer, dtype=torch.bfloat16
            )
            print("llm optimize done...")
        else:
            # Fall back to the regular optimize function
            model, optimizer = ipex.optimize(
                model=model, optimizer=optimizer, dtype=torch.bfloat16
            )
            print("ipex optimize done (commented out, getting nans...")
    total_loss = 0.0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()  # Reset gradients
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

        loss.backward()  # Compute gradients

        # Gradient accumulation
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(
            train_dataloader
        ) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if generate_while_training and (step + 1) % N == 0:
            model.eval()
            prompt = infer_prompt()
            model = model.merge_and_unload()
            generated_text = generate_text(model, prompt)
            print(f"\nGenerated text after batch {step+1}:\n{generated_text}")
            model.train()
        total_loss += loss.item()
        wandb.log(
            {
                "total_train_loss": total_loss,
            }
        )
    avg_train_loss = total_loss / len(train_dataloader)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))

    print(f"Average training loss: {avg_train_loss}")
    print(f"Training Perplexity: {train_perplexity}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss = outputs.loss

            total_val_loss += val_loss.item()
            wandb.log(
                {
                    "total_val_loss": total_val_loss,
                }
            )
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))

    print(f"Average validation loss: {avg_val_loss}")
    print(f"Validation Perplexity: {val_perplexity}")

    # Generate text at the end of each epoch
    prompt = infer_prompt()
    model = model.merge_and_unload()
    generated_text = generate_text(model, prompt)
    print(f"Generated text:\n{generated_text}")
    model = get_peft_model(model, config)
    model.train()
    wandb.log(
        {
            "Average training loss": avg_train_loss,
            "Training Perplexity": train_perplexity,
            "Average validation loss": avg_val_loss,
            "Validation Perplexity": val_perplexity,
        }
    )


# Save the model
model.save_pretrained(OUTPUT_DIR)
# model_artifact = wandb.Artifact(f"simpsons_model", type="model")
# model_artifact.add_file(OUTPUT_DIR)
# wandb.log_artifact(model_artifact)
