## Simpson's LLM on XPUs

Welcome to the 'Simpson's LLM XPU' repository where we finetune a Language Model (LLM) on Intel discrete GPUs (Intel® Data Center GPU Max 1550) to generate dialogues based on the 'Simpsons' dataset.

<img src=https://github.com/rahulunair/simpsons_llm_xpu/assets/786476/9b4f49a2-ead8-4d7b-b79f-d8b98cd75eeb width="30%">

The implementation leverages the original idea and exceptional work done by Replicate for the dataset prep. In case the [Replicate link](https://replicate.com/blog/fine-tune-llama-to-speak-like-homer-simpson) is unavailable, please refer to my forked [version](https://github.com/rahulunair/homerbot_errata) for guidelines on preparing the dataset. The preparation steps are laid out simply in a Jupyter notebook.

### Getting Started

To utilize this code, start by preparing the dataset as suggested in the Replicate blog.

### Finetuning

#### For a Single XPU Device:

```bash
python finetune.py
````


#### For a Multi-XPU Configuration (Multiple dGPUs) using oneCCL:

Regarding oneCCL, Intel oneAPI Collective Communications Library (oneCCL) is a library that provides routines needed for communication between devices in distributed systems. These routines are built with a focus on performance and provide efficient inter-node and intra-node communication, making them suitable for multi-node, multi-core CPUs, and accelerators. We use PyTorch bindings for oneCCL (`torch_ccl`) to do distributed training. We can install `torch_ccl` by using prebuilt weels from [here](https://github.com/intel/torch-ccl#install-prebuilt-wheel).

As we are using HuggingFace Trainer* object, we don't have to change the code in anyway, but execute the code using `mpi`.

First, set up the oneCCL environment variables by executing:

```bash
oneccl_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_path/env/setvars.sh
```

Then set these environment variables for MPI:

```bash
export MASTER_ADDR=127.0.0.1
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER=sockets
```

Then, execute the following command to initiate the finetuning process across multiple XPUs:

```bash
mpirun -n 4 python finetune.py    # uses 4 Intel Data Center GPU Max 1550
```
![image](https://github.com/rahulunair/simpsons_llm_xpu/assets/786476/93574ca5-3077-4807-99ce-724afd481885)

To debug oneccl backend, use this env variable:

```bash
export CCL_LOG_LEVEL=debug
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

*. we use a forked version of huggingface transformers, it can be found [here](https://github.com/rahulunair/transformers_xpu).
