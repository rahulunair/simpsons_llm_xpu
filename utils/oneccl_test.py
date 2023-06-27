import os
import torch
import intel_extension_for_pytorch
import torch.distributed as dist
import numpy as np
import time

def _time():
    if torch.xpu.is_available():
        torch.xpu.synchronize()
    return time.time()

# Setting up environment variables
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

# Defaults
size = 786432
num_iter = 10
warmup = 10
ops = ["broadcast", "all_reduce"]
data_types = ["bfloat16", "float32"]

if torch.xpu.is_available():
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend='ccl')
    device = "xpu"
else:
    dist.init_process_group(backend='gloo')
    device = "cpu"
  
print(f'Using device: {device}')

for op in ops:
    for data_type in data_types:
        N = 1
        while N <= size:
            for _ in range(warmup):
                data = torch.randn(N, dtype=torch.bfloat16 if data_type == "bfloat16" else torch.float32, device=device)
                with torch.no_grad():
                    if op == "broadcast":
                        dist.broadcast(data, 0)
                    else:
                        dist.all_reduce(data)

            elapsed = []
            for _ in range(num_iter):
                data = torch.randn(N, dtype=torch.bfloat16 if data_type == "bfloat16" else torch.float32, device=device)
                t = _time()
                with torch.no_grad():
                    if op == "broadcast":
                        dist.broadcast(data, 0)
                    else:
                        dist.all_reduce(data)
                elapsed.append((_time() - t) * 1e6)

            if N == size:
                break
            N = 2 * N
            if N > size:
                N = size

        print(f'{op} operation was successful on {device} with {data_type}.')
dist.destroy_process_group()
