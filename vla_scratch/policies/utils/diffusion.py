import torch
from typing import TypeVar

T = TypeVar("T", bound=torch.Tensor)


def repeat_batch(t: T, repeat_times: int) -> T:
    """Repeat a tensor along a new leading dimension then flatten."""
    return t.expand(repeat_times, *t.shape).reshape(-1, *t.shape[1:])


def build_beta_time_dist(
    alpha: float,
    beta: float,
    device: torch.device | str,
) -> torch.distributions.Distribution:
    """Construct a Beta distribution on the training device."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(beta_t, alpha_t)


def sample_clamped_time(
    time_dist: torch.distributions.Distribution,
    shape: torch.Size,
) -> torch.Tensor:
    """Sample diffusion timesteps with a small clamp to avoid numerical issues."""
    return time_dist.sample(shape) * 0.999 + 0.001


def sample_noise(shape, device, dtype):
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=dtype,
        device=device,
    )
