import torch
from typing import Iterable
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
import jaxtyping as at


def get_beta_dist(
    alpha: float, beta: float, device
) -> torch.distributions.Distribution:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(beta_t, alpha_t)
    return dist


def sample_time(
    time_dist: torch.distributions.Distribution, bsize: torch.Size
) -> at.Float[torch.Tensor, "b"]:
    return time_dist.sample(bsize) * 0.999 + 0.001


def sample_noise(shape, device, dtype):
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=dtype,
        device=device,
    )
