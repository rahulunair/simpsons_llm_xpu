### simpsons_llm_xpu

Finetune an LLM on intel discrete GPUs to generate dialogues based on the simpsons dataset

This is a implementation of the awesome work Replicate done, but for Intel dGPUS: https://replicate.com/blog/fine-tune-llama-to-speak-like-homer-simpson

#### How to use?

Prepare the dataset as suggested in the replicate blog and then:

On a single XPU device:

```bash
python finetune.py
````

On a multi XPU configuration (multi dGPUs):

```bash
oneccl_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_path/env/setvars.sh
mpirun -n 2 python finetune.py
```

Once finetuning is done,to check the model use:

```bash
python inference.py --infer
```
