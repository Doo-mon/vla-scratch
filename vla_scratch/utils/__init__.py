import os

import torch
import torch.distributed as dist

local_rank = 0
global_rank = 0
world_size = 1


def setup_dist():
    """
    Initialize DDP process group
    """
    global local_rank, global_rank, world_size
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    except ValueError:
        local_rank = 0
        global_rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

def print_with_rank(string):
    print(f"[Rank {global_rank}] {string}")

# def breakpoint_rank0():
#     if global_rank == 0:
#         breakpoint()
#     dist.barrier()

