import torch
import jaxtyping as at
from typing import Tuple


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

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# @torch.compile
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

