from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import os
import time
from typing import cast, Any, Optional, List, Tuple
from tqdm import tqdm
import wandb
import datetime
from setproctitle import setproctitle
import einops

import torch
import torch.nn.functional as F

import torch.distributed as dist
import torch.distributed.tensor
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp._fully_shard import (
    fully_shard,
    MixedPrecisionPolicy,
    FSDPModule,
    register_fsdp_forward_method,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

from tensordict import TensorDict

import hydra
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, JobConf, RunDir
from omegaconf import DictConfig, OmegaConf, MISSING

from vla_scratch.datasets.config import DataConfig

from vla_scratch.policies.config import PolicyConfig, create_policy
from vla_scratch.policies.pi.policy import PiPolicy
from vla_scratch.policies.pi.config import PiConfig
from vla_scratch.policies.utils import (
    get_beta_dist,
    sample_noise,
    sample_time,
)

from vla_scratch.helpers import (
    create_dataloaders,
    compute_sample_mse,
    expand_tensor,
    aggregate_tensordict,
)
from vla_scratch.transforms.data_types import DataSample

from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from vla_scratch.utils import setup_dist, print_with_rank

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["FSDP_ENABLE_BACKWARD_HOOKS"] = "1"

torch.set_float32_matmul_precision("high")


@dataclass
class WandbCfg:
    project: str = "vla-scratch"
    mode: str = "disabled"
    tags: List[str] = field(default_factory=lambda: [])


@dataclass
class TrainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi"},
            {"data": "libero-ipec"},
        ]
    )

    # data loader
    num_workers: int = 8
    prefetch_factor: int = 6
    split_seed: int = 42
    # optimization
    epochs: int = 20
    batch_size: int = 16
    grad_accum_steps: int = 1

    lr: float = 3e-6
    # Linearly ramp LR from 0 to lr over this many optimizer steps (0 disables)
    warmup_steps: int = 0
    # LR scheduling: start cosine anneal from the last N epochs
    # Set to 0 to disable cosine annealing
    cosine_anneal_epoch: int = 0

    betas: Tuple[float] = (0.99, 0.9999)
    eps: float = 1e-8
    weight_decay: float = 1e-4

    clip_grad_norm: float = 1.0
    num_noise_per_sample: int = 8
    num_noise_before_topk: int = 8
    detach_kv_cache: bool = False
    disp_loss_weight: float = 0.25

    # logging and evaluation
    exp_name: str = "pi-training"
    log_interval: int = 32
    eval_interval: int = 512
    save_interval: int = 1  # in epochs
    eval_fraction: float = 0.003
    eval_num_sample_steps: int = 10
    # data
    data: DataConfig = MISSING
    # model
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None
    load_optimizer: bool = True
    # wandb
    wandb: WandbCfg = field(default_factory=WandbCfg)

    # Hydra behavior overrides
    # - Do not change cwd automatically (job.chdir=False)
    # - Do not create .hydra subdir (output_subdir=null)
    # - Keep Hydra run dir as current directory (run.dir='.')
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            job=JobConf(chdir=False),
            output_subdir=None,
            run=RunDir(dir="."),
        )
    )


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig())


@hydra.main(config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    train_cfg = cast(TrainConfig, OmegaConf.to_object(cfg))

    if train_cfg.num_noise_before_topk < train_cfg.num_noise_per_sample:
        raise ValueError(
            "num_noise_before_topk must be >= num_noise_per_sample for top-k selection"
        )

    # Resolve checkpoint path (supports file or directory)
    if train_cfg.checkpoint_path is not None:
        cp = Path(train_cfg.checkpoint_path).resolve()
        # If a directory is provided, pick latest matching checkpoint
        if cp.is_dir():
            latest = find_latest_checkpoint(cp)
            train_cfg.checkpoint_path = latest if latest is not None else None
        else:
            train_cfg.checkpoint_path = cp

    # create timestamped output directory with exp_name
    now = datetime.datetime.now()
    date_stamp = now.strftime("%Y-%m-%d")
    time_stamp = now.strftime("%H-%M-%S")
    run_dir = Path("./outputs") / date_stamp / f"{time_stamp}-{train_cfg.exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    setproctitle(f"{time_stamp}-{train_cfg.exp_name}")

    assert (
        train_cfg.eval_interval % train_cfg.log_interval == 0
    ), "eval-interval must be multiple of log-interval"

    local_rank, global_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_with_rank("create dataloaders...")
    (
        dataloader,
        eval_dataloader,
        subtrain_dataloader,
    ) = create_dataloaders(train_cfg, world_size, global_rank)

    dummy_data: DataSample = next(iter(dataloader))[0]
    action_dim = dummy_data.action_chunk.actions.shape[-1]
    state_dim = dummy_data.observation.state.shape[-1]

    train_cfg.policy = cast(PiConfig, train_cfg.policy)
    train_cfg.policy.action_dim = action_dim
    train_cfg.policy.state_dim = state_dim

    print_with_rank("create model...")
    with torch.device(device):
        # with (torch.device(device), torch.dtype(torch.bfloat16)):
        model: PiPolicy = create_policy(train_cfg.policy)

    if world_size > 1:
        nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
        nnodes = world_size // nproc_per_node
        assert world_size == nproc_per_node * nnodes
        if nnodes > 1:
            mesh = dist.device_mesh.init_device_mesh(
                "cuda", (nnodes, nproc_per_node), mesh_dim_names=("node", "process")
            )
        else:
            # Single node: prefer a simple 1D mesh to avoid backend split_group requirements
            mesh = dist.device_mesh.init_device_mesh(
                "cuda", (world_size,), mesh_dim_names=("process",)
            )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        for layer in model.vlm_bridge.model.language_model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
            register_fsdp_forward_method(layer, model.vlm_bridge.layer_custom_forward_name)
        for block in model.action_expert.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp_policy)

        mp_policy_root = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        fully_shard(model, mesh=mesh, mp_policy=mp_policy_root)
        register_fsdp_forward_method(model, "encode_prefix")
        register_fsdp_forward_method(model, "predict_suffix")
        register_fsdp_forward_method(model, "sample_actions")

        def set_forward_backward_prefetch(
            layers: List[FSDPModule],
            num_to_forward_prefetch: int,
            num_to_backward_prefetch: int,
        ) -> None:
            for i, layer in enumerate(layers):
                if i >= len(layers) - num_to_forward_prefetch:
                    break
                layers_to_prefetch = [
                    layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
                ]
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)
            for i, layer in enumerate(layers):
                if i < num_to_backward_prefetch:
                    continue
                layers_to_prefetch = [
                    layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
                ]
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)

        set_forward_backward_prefetch(model.vlm_bridge.model.language_model.layers, 2, 2)
        set_forward_backward_prefetch(model.action_expert.blocks, 2, 2)

        model: FSDPModule | PiPolicy

    global_batch_size = train_cfg.batch_size * train_cfg.grad_accum_steps * world_size
    lr = min(float(train_cfg.lr * np.sqrt(global_batch_size)), 3e-4)
    base_lr = lr
    betas = tuple(np.pow(beta, global_batch_size) for beta in train_cfg.betas)
    eps = train_cfg.eps / np.sqrt(global_batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=train_cfg.weight_decay,
        foreach=False,
        fused=True,
    )

    if train_cfg.checkpoint_path is not None:
        load_checkpoint(
            model=model,
            checkpoint=train_cfg.checkpoint_path,
            global_rank=global_rank,
            optimizer=optimizer if train_cfg.load_optimizer else None,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for group in optimizer.param_groups:
        group["lr"] = lr

    if global_rank == 0:
        run = wandb.init(
            project=train_cfg.wandb.project,
            mode=train_cfg.wandb.mode,
            tags=train_cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))

        default_run_name = (
            f"{train_cfg.exp_name}-{datetime.datetime.now().strftime('%m-%d-%H-%M')}"
        )
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"

        # save config
        with open("train-cfg.yaml", "w") as f:
            OmegaConf.save(cfg, f)
        with open("policy-cfg.yaml", "w") as f:
            OmegaConf.save(train_cfg.policy, f)
        with open("data-cfg.yaml", "w") as f:
            OmegaConf.save(train_cfg.data, f)

    time_dist = get_beta_dist(1.0, 1.5, device=device)

    global_step = 0
    scheduler = None
    steps_per_epoch = max(1, len(dataloader) // train_cfg.grad_accum_steps)
    last_time = time.perf_counter()
    log_tds = []

    for epoch in range(train_cfg.epochs):
        # Initialize cosine annealing at the start of the first scheduled epoch
        if (
            scheduler is None
            and train_cfg.cosine_anneal_epoch > 0
            and epoch >= train_cfg.epochs - train_cfg.cosine_anneal_epoch
        ):
            # Number of remaining optimizer steps including this epoch
            remaining_epochs = train_cfg.epochs - epoch
            cosine_steps = max(1, steps_per_epoch * remaining_epochs)
            # Start cosine anneal from current LR down to 1e-7 by training end

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-7,
            )
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        pbar = range(steps_per_epoch)
        if local_rank == 0:
            pbar = tqdm(pbar, desc=f"Epoch {epoch+1}/{train_cfg.epochs}")

        model.train()
        data_loader_iter = iter(dataloader)
        for i in pbar:
            torch.cuda.nvtx.range_push("Zero Grad")
            if isinstance(model, FSDPModule):
                model.unshard()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.nvtx.range_pop()

            for _ in range(train_cfg.grad_accum_steps):
                torch.cuda.nvtx.range_push("DataLoader")
                data_sample, perf_dict = next(data_loader_iter)
                data_sample: DataSample = data_sample.to(device, non_blocking=True)
                perf_dict = perf_dict.to(device, non_blocking=True)
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Model Encode Prefix")
                _, prefix_pad_masks, prefix_key_values = model.encode_prefix(
                    observation=data_sample.observation,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Noise Sampling")
                actions = data_sample.action_chunk.actions
                action_horizon, action_dim = actions.shape[1], actions.shape[2]
                noise_candidates = sample_noise(
                    (
                        actions.shape[0],
                        train_cfg.num_noise_before_topk,
                        *actions.shape[1:],
                    ),
                    device,
                    dtype=actions.dtype,
                )
                action_flat = einops.rearrange(
                    actions, "b h d -> b 1 (h d)", h=action_horizon, d=action_dim
                )
                noise_flat = einops.rearrange(
                    noise_candidates,
                    "b k h d -> b k (h d)",
                    h=action_horizon,
                    d=action_dim,
                )
                if train_cfg.num_noise_before_topk == train_cfg.num_noise_per_sample:
                    selected_noise_flat = noise_flat
                else:
                    distances = torch.sum((noise_flat - action_flat) ** 2, dim=-1)
                    topk_indices = torch.topk(
                        distances,
                        k=train_cfg.num_noise_per_sample,
                        dim=1,
                        largest=False,
                    ).indices
                    selected_noise_flat = torch.gather(
                        noise_flat,
                        dim=1,
                        index=topk_indices.unsqueeze(-1).expand(
                            -1, -1, noise_flat.shape[-1]
                        ),
                    )
                # Match expand_tensor ordering (noise index first, then batch)
                selected_noise = einops.rearrange(
                    selected_noise_flat,
                    "b k (h d) -> (k b) h d",
                    h=action_horizon,
                    d=action_dim,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Expand Data Sample")
                data_sample = expand_tensor(data_sample, train_cfg.num_noise_per_sample)
                prefix_pad_masks = expand_tensor(
                    prefix_pad_masks, train_cfg.num_noise_per_sample
                )
                prefix_key_values = [
                    (
                        expand_tensor(k, train_cfg.num_noise_per_sample),
                        expand_tensor(v, train_cfg.num_noise_per_sample),
                    )
                    for k, v in prefix_key_values
                ]
                torch.cuda.nvtx.range_pop()

                if train_cfg.detach_kv_cache:
                    torch.cuda.nvtx.range_push("Detach KV Cache")
                    prefix_key_values = [
                        (k.detach(), v.detach()) for k, v in prefix_key_values
                    ]
                    torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Apply Noise")
                actions = data_sample.action_chunk.actions
                noise = selected_noise
                u_t = noise - actions
                timestep = sample_time(time_dist, data_sample.shape)
                noisy_actions = actions + timestep[:, None, None] * u_t
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Model Predict Suffix")
                v_t, disp_loss = model.predict_suffix(
                    state=data_sample.observation.state,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_key_values=prefix_key_values,
                    noisy_actions=noisy_actions,
                    time=timestep,
                )
                losses = F.mse_loss(u_t, v_t, reduction="none")
                flow_mse = losses.mean()
                loss = flow_mse + train_cfg.disp_loss_weight * disp_loss
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Loss Backward")
                (loss / train_cfg.grad_accum_steps / world_size).backward()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Optimizer Step")
            norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg.clip_grad_norm
            )
            # Linear warmup: ramp LR from 0 to base_lr over warmup_steps
            if train_cfg.warmup_steps and global_step < train_cfg.warmup_steps:
                warmup_factor = float(global_step + 1) / float(train_cfg.warmup_steps)
                warmup_lr = base_lr * max(0.0, min(1.0, warmup_factor))
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            torch.cuda.nvtx.range_pop()

            log_td = {}
            log_td["loss/flow_mse"] = flow_mse.detach()
            log_td["loss/disp_loss"] = disp_loss.detach()
            if isinstance(norm_before_clip, torch.distributed.tensor.DTensor):
                norm_before_clip = norm_before_clip.full_tensor()
            log_td["loss/grad_norm"] = norm_before_clip
            log_td = TensorDict(log_td, [])
            log_td["loading"] = perf_dict.mean(dim=0)

            log_tds.append(log_td)

            global_step += 1

            if global_step % train_cfg.log_interval == 0:
                # log metrics
                samples = (
                    global_step
                    * train_cfg.batch_size
                    * world_size
                    * train_cfg.grad_accum_steps
                )
                log_dict = {
                    "epoch": epoch,
                    "step": global_step,
                    "samples": samples,
                }
                log_dict["loss/lr"] = optimizer.param_groups[0]["lr"]

                # log fps
                this_time = time.perf_counter()
                elapsed_time = this_time - last_time
                last_time = this_time
                fps = (
                    train_cfg.batch_size
                    * train_cfg.grad_accum_steps
                    * train_cfg.log_interval
                    / elapsed_time
                )
                log_dict["perf/fps"] = fps
                log_dict["perf/fps.total"] = fps * world_size

                # log train stats (aggregate deterministically with a single all_reduce)
                log_td_mean: TensorDict = (
                    torch.stack(log_tds).mean(dim=0).type(torch.float32)
                )
                log_tds.clear()

                log_dict.update(aggregate_tensordict(log_td_mean, world_size))

                if global_step % train_cfg.eval_interval == 0:
                    # No pre-barrier; evaluation collectives below will synchronize
                    model.eval()
                    if isinstance(model, FSDPModule):
                        model.unshard()
                    for key, eval_dataloader in [
                        ("eval", eval_dataloader),
                        ("train", subtrain_dataloader),
                    ]:
                        eval_mse = compute_sample_mse(
                            model=model,
                            dataloader=eval_dataloader,
                            device=device,
                            num_sample_steps=train_cfg.eval_num_sample_steps,
                            local_rank=local_rank,
                        )
                        if world_size > 1:
                            dist.all_reduce(eval_mse, op=dist.ReduceOp.AVG)
                        log_dict[f"loss/sample_mse-{key}"] = eval_mse.item()
                    if isinstance(model, FSDPModule):
                        model.reshard()
                    model.train()

                log_string = "\n".join(
                    [
                        (
                            f"{key}={value:.8f}"
                            if isinstance(value, float)
                            else f"{key}={value}"
                        )
                        for key, value in log_dict.items()
                    ]
                )
                if local_rank == 0:
                    print(log_string)
                if global_rank == 0:
                    run.log(log_dict)

        if (
            (epoch + 1) % train_cfg.save_interval == 0
        ):
            save_checkpoint(model, optimizer, global_rank, f"checkpoint_{epoch+1}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
