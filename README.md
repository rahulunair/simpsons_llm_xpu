## Simpson's LLM on XPUs

Welcome to the 'Simpson's LLM XPU' repository where we finetune a Language Model (LLM) on Intel discrete GPUs to generate dialogues based on the 'Simpsons' dataset.

<img src=https://github.com/rahulunair/simpsons_llm_xpu/assets/786476/9b4f49a2-ead8-4d7b-b79f-d8b98cd75eeb width="30%">

The implementation leverages the exceptional work done by Replicate for the dataset prep. In case the [Replicate link](https://replicate.com/blog/fine-tune-llama-to-speak-like-homer-simpson) is unavailable, please refer to my forked [version](https://github.com/rahulunair/homerbot_errata) for guidelines on preparing the dataset. The preparation steps are laid out simply in a Jupyter notebook.

### Getting Started

To utilize this code, start by preparing the dataset as suggested in the Replicate blog.

### Finetuning

#### For a Single XPU Device:

```bash
python finetune.py
````

#### For a Multi-XPU Configuration (Multiple dGPUs):

First, set up the oneCCL environment variables by executing:

```bash
oneccl_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_path/env/setvars.sh
```

Then, execute the following command to initiate the finetuning process across multiple XPUs:

```bash
mpirun -n 2 python finetune.py
```

### Post Finetuning:

Once the finetuning is complete, you can test the model with the following command:

```bash
python inference.py --infer
```

### Literate Version of Finetuning

To get a better understanding of the Low-rank Option for finetuning Transformers (LORΛ) and the finetuning approach, I have added  a literate version of the finetune.py file as a Jupyter notebook - literate_finetune.ipynb. This version provides detailed explanations of each step and includes code snippets to provide a comprehensive understanding of the finetuning process.

By going through this literate version, I hope that you can gain insights into the workings of LORΛ, how it interacts with the training process, and how you can utilize Intel GPUs for efficient finetuning. This is especially beneficial for practitioners new to language model finetuning, or those looking to gain a deeper understanding of the process.

Happy Finetuning!
