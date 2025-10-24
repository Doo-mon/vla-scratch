from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import time
from typing import cast
from tqdm import tqdm
import wandb
import datetime
from setproctitle import setproctitle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

from tensordict import TensorDict

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:
    # dataset and loader
    data_root: Path = Path("datasets/libero_spatial")
    num_workers: int = 4
    split_seed: int = 42
    # optimization
    epochs: int = 5
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 3e-5
    weight_decay: float = 1e-4
    optim_eps: float = 1e-8
    clip_grad_norm: float = 1.0
    # model
    action_expert_variant: str = "300m"
    paligemma_checkpoint: str | None = None
    action_horizon: int = 30
    state_history: int = 10
    # logging and evaluation
    log_interval: int = 20
    eval_interval: int = 40
    eval_fraction: float = 0.1
    eval_num_sample_steps: int = 10
    save_path: Path | None = None
    # wandb
    exp_name: str = "pi-training"
    wandb_project: str = "vla-scratch"
    wandb_mode: str = "disabled"


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig())


from vla_scratch.datasets.libero import LiberoDataset
from vla_scratch.datasets.data_types import DataSample
from vla_scratch.policies.pi.config import PiConfig
from vla_scratch.policies.pi.policy import PiPolicy
from vla_scratch.policies.pi.utils import *


def load_checkpoint(model: PiPolicy, checkpoint: str) -> None:
    """Load a Paligemma pretrained checkpoint into the PI0 backbone."""

    from transformers import AutoModel

    ckpt_path = Path(checkpoint)
    source = str(ckpt_path) if ckpt_path.exists() else checkpoint

    if ckpt_path.is_file():
        if ckpt_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "safetensors is required to load Paligemma checkpoints stored as .safetensors"
                ) from exc
            state_dict = load_file(source)
        else:
            state_dict = torch.load(source, map_location="cpu")
    else:
        pretrained = AutoModel.from_pretrained(
            source,
            dtype=torch.float32,
            device_map=None,
        )
        state_dict = pretrained.state_dict()
        del pretrained

    missing, unexpected = model.paligemma.model.load_state_dict(
        state_dict, strict=False
    )

    if missing:
        print(f"Paligemma checkpoint loaded with missing keys: {missing}")
    if unexpected:
        print(f"Paligemma checkpoint loaded with unexpected keys: {unexpected}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    global_rank: int,
    filename: str,
):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state_dict, optim_state_dict = get_state_dict(
        model,
        optimizers=optimizer,
        options=options,
    )

    if global_rank == 0 and train_cfg.save_path is not None:
        train_cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        full_state_dict = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
        }
        torch.save(full_state_dict, filename)
        print(f"Saved checkpoint to {filename}")


def setup_ddp():
    """
    Initialize DDP process group
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", device_id=local_rank)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    except ValueError:
        local_rank = 0
        global_rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["FSDP_ENABLE_BACKWARD_HOOKS"] = "1"
    return local_rank, global_rank, world_size


def create_dataloaders(train_cfg: TrainConfig, world_size: int, global_rank: int):
    from transformers import PaliGemmaProcessor

    model_id = "google/paligemma-3b-mix-224"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    if not (0.0 < train_cfg.eval_fraction < 1.0):
        raise ValueError("eval_fraction must be within (0, 1).")

    full_dataset = LiberoDataset(
        train_cfg.data_root,
        tokenizer=tokenizer,
        action_horizon=train_cfg.action_horizon,
        state_history=train_cfg.state_history,
        max_tokens=128,
    )
    from vla_scratch.datasets.base import TransformedDataset
    from vla_scratch.datasets.transforms import LiberoProprio, ToTensorClass, Normalize

    transforms = [
        LiberoProprio(),
        Normalize(stats_file=Path("normalization_stats/libero_proprio_stats.npz")),
        ToTensorClass(),
    ]
    full_dataset = TransformedDataset(full_dataset, transforms)

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
    subtrain_size = min(subtrain_size, train_size)
    subtrain_generator = torch.Generator().manual_seed(train_cfg.split_seed + 1)
    subtrain_indices = torch.randperm(train_size, generator=subtrain_generator)[
        :subtrain_size
    ].tolist()
    subtrain_dataset = torch.utils.data.Subset(train_dataset, subtrain_indices)

    def _create_dataloader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=shuffle,
                drop_last=shuffle,
            )
        else:
            sampler = None

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=train_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=torch.stack,
        )
        if train_cfg.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        else:
            loader_kwargs["shuffle"] = shuffle

        return DataLoader(dataset, **loader_kwargs)

    dataloader = _create_dataloader(
        train_dataset, shuffle=True, batch_size=train_cfg.batch_size
    )
    eval_dataloader = _create_dataloader(eval_dataset, shuffle=False, batch_size=32)
    subtrain_dataloader = _create_dataloader(
        subtrain_dataset, shuffle=False, batch_size=32
    )
    return (
        dataloader,
        eval_dataloader,
        subtrain_dataloader,
    )


@torch.inference_mode()
def compute_sample_mse(
    model: PiPolicy,
    dataloader: DataLoader,
    device: torch.device,
    num_sample_steps: int,
    global_rank: int,
) -> torch.Tensor:
    squared_errors = []

    pbar = range(len(dataloader))
    if global_rank == 0:
        pbar = tqdm(pbar, desc=f"Evaluating sample MSE")
    for i in pbar:
        batch = next(iter(dataloader))
        batch: DataSample = batch.to(device)
        predicted_actions = model.forward(
            part="sample",
            observation=batch.observation,
            num_steps=num_sample_steps,
        )
        target_actions = batch.action.actions

        squared_error = F.mse_loss(
            predicted_actions,
            target_actions,
            reduction="none",
        )
        squared_errors.append(squared_error.mean())

    return torch.stack(squared_errors).mean()

def get_fsdp_wrap_policy():
    """Get the FSDP auto wrap policy for PaliGemma model.
    
    Returns:
        FSDP transformer auto wrap policy
    """
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools
    from vla_scratch.policies.modules.dit import DecoderBlock
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
    
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=[
            DecoderBlock,
            # GemmaDecoderLayer,
        ],
    )
    return transformer_wrap_policy

@hydra.main(config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    train_cfg = cast(TrainConfig, OmegaConf.to_object(cfg))
    train_cfg.data_root = Path(to_absolute_path(str(train_cfg.data_root)))
    if train_cfg.save_path is not None:
        train_cfg.save_path = Path(to_absolute_path(str(train_cfg.save_path)))

    assert (
        train_cfg.eval_interval % train_cfg.log_interval == 0
    ), "eval-interval must be multiple of log-interval"

    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    (
        dataloader,
        eval_dataloader,
        subtrain_dataloader,
    ) = create_dataloaders(train_cfg, world_size, global_rank)

    dummy_data: DataSample = next(iter(dataloader))
    action_dim = dummy_data.action.actions.shape[-1]
    state_dim = dummy_data.observation.state.shape[-1]

    base_config = PiConfig(
        action_expert_variant=train_cfg.action_expert_variant,
        action_dim=action_dim,
        action_horizon=train_cfg.action_horizon,
        state_dim=state_dim,
        state_history=train_cfg.state_history,
    )
    with torch.device(device):
        model = PiPolicy(base_config)

    if local_rank == 0 and train_cfg.paligemma_checkpoint:
        load_checkpoint(model, train_cfg.paligemma_checkpoint)

    if world_size > 1:
        # TODO: currently change to float32 for reduce type will make training very slow
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
            cast_forward_inputs=True,
            cast_root_forward_inputs=True,
        )

        nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
        nnodes = world_size // nproc_per_node
        assert world_size == nproc_per_node * nnodes
        mesh = dist.device_mesh.init_device_mesh("cuda", (nnodes, nproc_per_node))
        # get the process group within a node
        pg = mesh.get_group(mesh_dim=1)

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh,
            cpu_offload=False,
            mixed_precision=mixed_precision,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
            sync_module_states=True,
            # auto_wrap_policy=get_fsdp_wrap_policy(),
        )
        print(f"Wrapped model in FSDP: global_rank={global_rank}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        eps=train_cfg.optim_eps,
        foreach=False,
        fused=True,
    )

    if global_rank == 0:
        run = wandb.init(
            project=train_cfg.wandb_project,
            mode=train_cfg.wandb_mode,
        )
        run.config.update(OmegaConf.to_container(cfg))

        default_run_name = (
            f"{train_cfg.exp_name}-{datetime.datetime.now().strftime('%m-%d-%H-%M')}"
        )
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

    time_dist = get_beta_dist(1.0, 1.5, device=device)

    global_step = 0
    last_time = time.perf_counter()
    for epoch in range(train_cfg.epochs):
        data_loader_iter = iter(dataloader)
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        log_tds = []

        pbar = range(len(dataloader) // train_cfg.grad_accum_steps)
        if global_rank == 0:
            pbar = tqdm(pbar, desc=f"Epoch {epoch+1}/{train_cfg.epochs}")

        model.train()
        for i in pbar:
            torch.cuda.nvtx.range_push("Zero Grad")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.nvtx.range_pop()

            for _ in range(train_cfg.grad_accum_steps):
                torch.cuda.nvtx.range_push("DataLoader")
                data_sample = next(data_loader_iter)
                data_sample: DataSample = data_sample.to(device)
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Model Encode Prefix")
                _, prefix_pad_masks, prefix_key_values = model.forward(
                    part="prefix",
                    observation=data_sample.observation,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Noise Sampling")
                actions = data_sample.action.actions
                noise = sample_noise(actions.shape, device, dtype=actions.dtype)
                u_t = noise - actions
                timestep = sample_time(time_dist, data_sample.batch_size)
                noisy_actions = actions + timestep[:, None, None] * u_t
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Model Predict Suffix")
                v_t = model.forward(
                    part="suffix",
                    state=data_sample.observation.state,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_key_values=prefix_key_values,
                    noisy_actions=noisy_actions,
                    time=timestep,
                )
                losses = F.mse_loss(u_t.type_as(v_t), v_t, reduction="none")
                loss = losses.mean()
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Loss Backward")
                (loss / math.sqrt(train_cfg.grad_accum_steps)).backward()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Optimizer Step")
            norm_before_clip = clip_grad_norm_(
                model.parameters(),
                max_norm=train_cfg.clip_grad_norm,
                norm_type=2.0,
                pg=pg if world_size > 1 else None,
            )
            optimizer.step()
            torch.cuda.nvtx.range_pop()

            data_stats_td = {
                "data/observation.state.mean": data_sample.observation.state.mean().detach(),
                "data/observation.state.std": data_sample.observation.state.std().detach(),
                "data/observation.state.min": data_sample.observation.state.min().detach(),
                "data/observation.state.max": data_sample.observation.state.max().detach(),
                "data/observation.images.mean": data_sample.observation.images.mean().detach(),
                "data/observation.images.std": data_sample.observation.images.std().detach(),
                "data/observation.images.min": data_sample.observation.images.min().detach(),
                "data/observation.images.max": data_sample.observation.images.max().detach(),
                "data/action.mean": data_sample.action.actions.mean().detach(),
                "data/action.std": data_sample.action.actions.std().detach(),
                "data/action.min": data_sample.action.actions.min().detach(),
                "data/action.max": data_sample.action.actions.max().detach(),
            }
            log_td = {}
            log_td["loss/flow_mse"] = loss.detach()
            log_td["loss/grad_norm"] = norm_before_clip.detach()
            # log_td.update(data_stats_td)

            log_tds.append(TensorDict(log_td, []))

            global_step += 1
            if global_step % train_cfg.log_interval == 0:
                # log metrics
                log_dict = {
                    "epoch": epoch,
                    "step": global_step,
                    "samples": global_step
                    * train_cfg.batch_size
                    * world_size
                    * train_cfg.grad_accum_steps,
                }
                log_dict["loss/lr"] = optimizer.param_groups[0]["lr"]

                # log fps
                this_time = time.perf_counter()
                elapsed_time = this_time - last_time
                last_time = this_time
                fps = (
                    train_cfg.batch_size
                    * train_cfg.grad_accum_steps
                    / elapsed_time
                )
                log_dict["perf/fps"] = fps
                log_dict["perf/fps.total"] = fps * world_size

                # log train stats
                log_td_stack: TensorDict = torch.stack(log_tds, dim=0)
                if world_size > 1:
                    for tensor in log_td_stack.values(leaves_only=True):
                        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                log_tds.clear()
                log_td_mean = log_td_stack.type(torch.float32).mean(dim=0)
                log_dict.update(log_td_mean.cpu().numpy())

                if global_step % train_cfg.eval_interval == 0:
                    if world_size > 1:
                        dist.barrier()
                    model.eval()
                    subtrain_mse = compute_sample_mse(
                        model=model,
                        dataloader=subtrain_dataloader,
                        device=device,
                        num_sample_steps=train_cfg.eval_num_sample_steps,
                        global_rank=global_rank,
                    )
                    eval_mse = compute_sample_mse(
                        model=model,
                        dataloader=eval_dataloader,
                        device=device,
                        num_sample_steps=train_cfg.eval_num_sample_steps,
                        global_rank=global_rank,
                    )
                    if world_size > 1:
                        dist.all_reduce(subtrain_mse, op=dist.ReduceOp.AVG)
                        dist.all_reduce(eval_mse, op=dist.ReduceOp.AVG)
                        dist.barrier()
                    log_dict["loss/sample_mse-train"] = subtrain_mse.item()
                    log_dict["loss/sample_mse-eval"] = eval_mse.item()
                    model.train()

                if global_rank == 0:
                    run.log(log_dict)
                    log_string = "\n".join(
                        [
                            (
                                f"{key}={value:.6f}"
                                if isinstance(value, float)
                                else f"{key}={value}"
                            )
                            for key, value in log_dict.items()
                        ]
                    )
                    print(log_string)

        save_checkpoint(
            model, optimizer, train_cfg, global_rank, f"checkpoint_{epoch+1}.pth"
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
