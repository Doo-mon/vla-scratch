from __future__ import annotations

import datetime
import logging
import os
from typing import TYPE_CHECKING, Any, Mapping, Set
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.tensor import DTensor

from vla_scratch.helpers.data import create_dataset
from vla_scratch.utils.dataloader import DistributedRankAwareBatchSampler

if TYPE_CHECKING:
    from tensordict import TensorDict
    from scripts.train_policy import TrainConfig
    from vla_scratch.policies.base import BasePolicy
    from vla_scratch.transforms.data_types import DataSample

logger = logging.getLogger(__name__)

local_rank = 0
global_rank = 0
world_size = 1


def setup_dist():
    """Initialize DDP process group using env:// init and optionally build a device mesh."""
    global local_rank, global_rank, world_size
    mesh = None
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        timeout_sec = int(os.environ.get("TORCH_DDP_TIMEOUT_SEC", 600))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_sec),
        )
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size > 1:
            nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
            nnodes = world_size // nproc_per_node
            assert world_size == nproc_per_node * nnodes
            if nnodes > 1:
                mesh = dist.device_mesh.init_device_mesh(
                    "cuda",
                    (nnodes, nproc_per_node),
                    mesh_dim_names=("node", "process"),
                )
            else:
                mesh = dist.device_mesh.init_device_mesh(
                    "cuda",
                    (world_size,),
                    mesh_dim_names=("process",),
                )
        print(f"Initialized DDP: rank {global_rank}/{world_size}, local rank {local_rank}")
    except ValueError:
        local_rank = 0
        global_rank = 0
        world_size = 1
        mesh = None
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size, mesh


def print_with_rank(string: str) -> None:
    print(f"[Rank {global_rank}] {string}")


def _create_dataloader(
    *,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    train_cfg: "TrainConfig",
    world_size: int,
    global_rank: int,
) -> DataLoader:
    base_sampler = getattr(dataset, "sampler", None)
    distributed_batch_sampler = None

    if world_size > 1:
        if base_sampler is not None:
            distributed_batch_sampler = DistributedRankAwareBatchSampler(
                base_sampler,
                batch_size=batch_size,
                drop_last=shuffle,
                num_replicas=world_size,
                rank=global_rank,
            )
            sampler = None
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=shuffle,
                drop_last=shuffle,
            )
    else:
        sampler = base_sampler

    def collate_fn(batch):
        return tuple(torch.stack(items) for items in zip(*batch))

    loader_kwargs = dict(
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    if train_cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = train_cfg.prefetch_factor
    if distributed_batch_sampler is not None:
        loader_kwargs["batch_sampler"] = distributed_batch_sampler
    elif sampler is not None:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle

    return DataLoader(dataset, **loader_kwargs)


def create_dataloaders(
    train_cfg: "TrainConfig",
    world_size: int,
    global_rank: int,
    *,
    add_noise: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_cfg.data.action_horizon = train_cfg.policy.action_horizon
    train_cfg.data.state_history = train_cfg.policy.state_history

    full_dataset = create_dataset(
        train_cfg.data,
        train_cfg.policy,
        add_noise=add_noise,
    )

    if not (0.0 < train_cfg.eval_fraction < 1.0):
        raise ValueError("eval_fraction must be within (0, 1).")

    total_samples = len(full_dataset)
    eval_size = max(1, int(total_samples * train_cfg.eval_fraction))
    train_size = total_samples - eval_size

    split_generator = torch.Generator().manual_seed(train_cfg.split_seed)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, eval_size],
        generator=split_generator,
    )
    train_size = len(train_dataset)

    subtrain_size = max(1, int(train_size * train_cfg.eval_fraction))
    subtrain_generator = torch.Generator().manual_seed(train_cfg.split_seed + 1)
    subtrain_indices = torch.randperm(train_size, generator=subtrain_generator)[
        :subtrain_size
    ].tolist()
    subtrain_dataset = torch.utils.data.Subset(train_dataset, subtrain_indices)

    dataloader = _create_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=train_cfg.batch_size,
        train_cfg=train_cfg,
        world_size=world_size,
        global_rank=global_rank,
    )
    eval_dataloader = _create_dataloader(
        dataset=eval_dataset,
        shuffle=False,
        batch_size=32,
        train_cfg=train_cfg,
        world_size=world_size,
        global_rank=global_rank,
    )
    subtrain_dataloader = _create_dataloader(
        dataset=subtrain_dataset,
        shuffle=False,
        batch_size=32,
        train_cfg=train_cfg,
        world_size=world_size,
        global_rank=global_rank,
    )

    return (
        dataloader,
        eval_dataloader,
        subtrain_dataloader,
    )


def build_param_lr_groups(
    model: torch.nn.Module,
    lr_cfg: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Create optimizer parameter groups from a learning-rate mapping."""
    if not lr_cfg:
        return [{"params": list(model.parameters()), "name": "base"}]

    base_lr = lr_cfg.get("base")
    used_params: Set[int] = set()
    param_groups: list[dict[str, Any]] = []

    for module_path, lr in lr_cfg.items():
        if module_path == "base":
            continue
        try:
            module = model
            for attr in module_path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            logger.warning(
                "Learning rate config references missing module path '%s'; skipping.",
                module_path,
            )
            continue

        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            continue

        param_groups.append({"params": params, "lr": float(lr), "name": module_path})
        used_params.update(id(p) for p in params)

    remaining_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in used_params
    ]
    if remaining_params:
        base_group: dict[str, Any] = {"params": remaining_params, "name": "base"}
        if base_lr is not None:
            base_group["lr"] = float(base_lr)
        param_groups.append(base_group)

    return param_groups


@torch.inference_mode()
def compute_sample_mse(
    model: "BasePolicy",
    dataloader: DataLoader,
    device: torch.device,
    num_sample_steps: int,
    local_rank: int,
) -> torch.Tensor:
    squared_errors = []

    pbar = tqdm(
        range(len(dataloader)),
        desc="Evaluating sample MSE",
        disable=local_rank != 0,
    )
    dataloader_iter = iter(dataloader)
    for _ in pbar:
        batch, _ = next(dataloader_iter)
        batch: "DataSample" = batch.to(device)
        predicted_actions = model.sample_actions(
            observation=batch.observation,
            num_steps=num_sample_steps,
        )
        target_actions = batch.action_chunk.actions

        squared_error = F.mse_loss(
            predicted_actions,
            target_actions,
            reduction="none",
        )
        squared_errors.append(squared_error.mean())

    return torch.stack(squared_errors).mean()
def aggregate_tensordict(td: "TensorDict", world_size: int) -> dict[str, float]:
    flat_td = td.flatten_keys(separator="/")
    if world_size <= 1:
        return flat_td.to_dict(convert_tensors=True)
    flat_dict = flat_td.to_dict()
    keys_sorted = sorted(flat_dict.keys())

    vec = torch.stack(
        [flat_dict[k].detach().reshape(1) for k in keys_sorted],
        dim=0,
    ).squeeze(-1)

    dist.all_reduce(vec, op=dist.ReduceOp.AVG)

    agg_values = vec.detach().cpu().tolist()
    return {k: float(agg_values[i]) for i, k in enumerate(keys_sorted)}


def _add_dtype_bytes(
    dtype_bytes: dict[torch.dtype, int], dtype: torch.dtype, num_bytes: int
) -> None:
    dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + num_bytes


def _accumulate_tensor_stats(
    tensors: list[torch.Tensor | DTensor],
) -> tuple[int, dict[torch.dtype, int]]:
    """Return total bytes and float-dtype bytes for a list of tensors."""
    total_bytes_local = 0
    total_bytes_global = 0
    float_dtype_bytes: dict[torch.dtype, int] = {}
    for tensor in tensors:
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
            num_bytes = tensor.numel() * tensor.element_size()
            total_bytes_local += num_bytes
            total_bytes_global += num_bytes * dist.get_world_size()
        else:
            num_bytes = tensor.numel() * tensor.element_size()
            total_bytes_local += num_bytes
            total_bytes_global += num_bytes
        if tensor.is_floating_point():
            _add_dtype_bytes(float_dtype_bytes, tensor.dtype, num_bytes)
    return total_bytes_local, total_bytes_global, float_dtype_bytes


def _accumulate_state_stats(state: Any) -> tuple[int, dict[torch.dtype, int]]:
    """Recursively return total bytes and float-dtype bytes for optimizer state."""
    if isinstance(state, (torch.Tensor, DTensor)):
        local_tensor = state.to_local() if isinstance(state, DTensor) else state
        num_bytes = local_tensor.numel() * local_tensor.element_size()
        float_dtype_bytes: dict[torch.dtype, int] = {}
        if local_tensor.is_floating_point():
            _add_dtype_bytes(float_dtype_bytes, local_tensor.dtype, num_bytes)
        return num_bytes, float_dtype_bytes

    if isinstance(state, Mapping):
        total_bytes = 0
        float_dtype_bytes: dict[torch.dtype, int] = {}
        for value in state.values():
            child_bytes, child_dtype_bytes = _accumulate_state_stats(value)
            total_bytes += child_bytes
            for dtype, bytes_ in child_dtype_bytes.items():
                _add_dtype_bytes(float_dtype_bytes, dtype, bytes_)
        return total_bytes, float_dtype_bytes

    if isinstance(state, (list, tuple)):
        total_bytes = 0
        float_dtype_bytes: dict[torch.dtype, int] = {}
        for value in state:
            child_bytes, child_dtype_bytes = _accumulate_state_stats(value)
            total_bytes += child_bytes
            for dtype, bytes_ in child_dtype_bytes.items():
                _add_dtype_bytes(float_dtype_bytes, dtype, bytes_)
        return total_bytes, float_dtype_bytes

    return 0, {}


def _format_dtype_bytes(dtype_bytes: dict[torch.dtype, int]) -> str:
    if not dtype_bytes:
        return "none"
    parts = [
        f"{dtype.__str__().replace('torch.', '')}={bytes_ / (1024**2):.2f}MB"
        for dtype, bytes_ in sorted(dtype_bytes.items(), key=lambda x: x[0].__str__())
    ]
    return ", ".join(parts)


def log_model_state_sizes(
    policy: "BasePolicy", optimizer: torch.optim.Optimizer
) -> None:
    """Print local-shard sizes (MB) for params, buffers, grads, and optimizer state."""
    params = list(policy.parameters())
    buffers = list(policy.buffers())
    grads = [p.grad for p in params if p.grad is not None]


    param_bytes_local, param_bytes_global, param_dtype_bytes = _accumulate_tensor_stats(
        params
    )
    buffer_bytes_local, buffer_bytes_global, buffer_dtype_bytes = _accumulate_tensor_stats(
        buffers
    )
    grad_bytes_local, grad_bytes_global, grad_dtype_bytes = _accumulate_tensor_stats(
        grads
    )
    optim_bytes_local = 0
    optim_dtype_bytes: dict[torch.dtype, int] = {}
    for state in optimizer.state.values():
        state_bytes, state_dtype_bytes = _accumulate_state_stats(state)
        optim_bytes_local += state_bytes
        for dtype, bytes_ in state_dtype_bytes.items():
            _add_dtype_bytes(optim_dtype_bytes, dtype, bytes_)

    def to_mb(num_bytes: int) -> float:
        return num_bytes / (1024**2)

    total_bytes_local = (
        param_bytes_local
        + buffer_bytes_local
        + grad_bytes_local
        + optim_bytes_local
    )
    total_bytes_global = (
        param_bytes_global
        + buffer_bytes_global
        + grad_bytes_global
    )
    # global means full model, not sum over all ranks
    msg = (
        "Tensor sizes (MB): "
        f"params[local={to_mb(param_bytes_local):.2f}, "
        f"global={to_mb(param_bytes_global):.2f}], "
        f"buffers[local={to_mb(buffer_bytes_local):.2f}, "
        f"global={to_mb(buffer_bytes_global):.2f}], "
        f"grads[local={to_mb(grad_bytes_local):.2f}, "
        f"global={to_mb(grad_bytes_global):.2f}], "
        f"optim_state[local={to_mb(optim_bytes_local):.2f}] | "
        f"Total[local={to_mb(total_bytes_local):.2f}, "
        f"global={to_mb(total_bytes_global):.2f}]; "
        "float dtypes: "
        f"params[{_format_dtype_bytes(param_dtype_bytes)}], "
        f"buffers[{_format_dtype_bytes(buffer_dtype_bytes)}], "
        f"grads[{_format_dtype_bytes(grad_dtype_bytes)}], "
        f"optim_state[{_format_dtype_bytes(optim_dtype_bytes)}]"
    )
    print_with_rank(msg)
