from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import os
import time
from typing import Any, cast, Optional, List, Tuple
from tqdm import tqdm
import wandb
import datetime
from setproctitle import setproctitle
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.distributed.tensor
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp._fully_shard import (
    fully_shard,
    MixedPrecisionPolicy,
    FSDPModule,
    register_fsdp_forward_method,
)
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from tensordict import TensorDict

import hydra
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, JobConf, RunDir
from omegaconf import DictConfig, OmegaConf, MISSING

from vla_scratch.datasets.config import DataConfig, create_dataset
from vla_scratch.policies.config import PolicyConfig, create_policy
from vla_scratch.datasets.data_types import DataSample

from vla_scratch.policies.pi.policy import PiPolicy
from vla_scratch.utils import setup_dist, print_with_rank
from vla_scratch.policies.utils import (
    get_beta_dist,
    sample_noise,
    sample_time,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FSDP_ENABLE_BACKWARD_HOOKS"] = "1"

torch.set_float32_matmul_precision("high")


@dataclass
class WandbCfg:
    project: str = "vla-scratch"
    mode: str = "disabled"


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
    split_seed: int = 42
    # optimization
    epochs: int = 20
    batch_size: int = 16
    grad_accum_steps: int = 1

    lr: float = 3e-6
    betas: Tuple[float] = (0.99, 0.9999)
    eps: float = 1e-8
    weight_decay: float = 1e-4

    clip_grad_norm: float = 1.0
    num_noise_per_sample: int = 8
    # logging and evaluation
    exp_name: str = "pi-training"
    log_interval: int = 32
    eval_interval: int = 512
    eval_fraction: float = 0.01
    eval_num_sample_steps: int = 10

    # data
    data: DataConfig = MISSING
    # model
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None
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


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: str,
    global_rank: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    # Only rank 0 reads from disk; others pass empty dict and receive via broadcast
    print_with_rank("Loading state dict from disk")
    if global_rank == 0:
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        model_sd = state_dict.get("model", {})
        optim_sd = state_dict.get("optimizer", {})
    else:
        model_sd = {}
        optim_sd = {}
    print_with_rank("Loaded state dict from disk!")

    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    set_model_state_dict(model=model, model_state_dict=model_sd, options=options)
    print_with_rank("Set model state dict!")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Always call optimizer load on all ranks to keep collectives in sync
    if optimizer is not None:
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=optim_sd,
            options=options,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_rank: int,
    filename: str,
):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state_dict, optim_state_dict = get_state_dict(
        model,
        optimizers=optimizer,
        options=options,
    )

    if global_rank == 0:
        full_state_dict = {
            "model": model_state_dict,
            # "optimizer": optim_state_dict,
        }
        torch.save(full_state_dict, filename)
        print_with_rank(f"Saved checkpoint to {filename}")




def create_dataloaders(train_cfg: TrainConfig, world_size: int, global_rank: int):
    train_cfg.data.action_horizon = train_cfg.policy.action_horizon
    train_cfg.data.state_history = train_cfg.policy.state_history

    full_dataset = create_dataset(
        train_cfg.data,
        train_cfg.policy,
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

        def collate_fn(batch):
            return tuple(torch.stack(items) for items in zip(*batch))

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=train_cfg.num_workers,
            persistent_workers=train_cfg.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
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
    dataloader_iter = iter(dataloader)
    if isinstance(model, FSDPModule):
        model.unshard()
    for i in pbar:
        batch, _ = next(dataloader_iter)
        batch: DataSample = batch.to(device)
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
    if isinstance(model, FSDPModule):
        model.reshard()

    return torch.stack(squared_errors).mean()


@hydra.main(config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    train_cfg = cast(TrainConfig, OmegaConf.to_object(cfg))

    # convert train_cfg.checkpoint_path to absolute path so after chdir 
    if train_cfg.checkpoint_path is not None:
        train_cfg.checkpoint_path = Path(train_cfg.checkpoint_path).resolve()

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

    train_cfg.policy.action_dim = action_dim
    train_cfg.policy.state_dim = state_dim

    print_with_rank("create model...")
    with torch.device(device):
        # with (torch.device(device), torch.dtype(torch.bfloat16)):
        model: PiPolicy = create_policy(train_cfg.policy)

    # Defer loading until after FSDP wrap so that DCP can shard+broadcast from rank 0

    if world_size > 1:
        nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
        nnodes = world_size // nproc_per_node
        assert world_size == nproc_per_node * nnodes
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (nnodes, nproc_per_node), mesh_dim_names=("node", "process")
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        for layer in model.paligemma.language_model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
            register_fsdp_forward_method(layer, model.gemma_custom_forward_name)
        for block in model.gemma_expert.blocks:
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

        set_forward_backward_prefetch(model.paligemma.language_model.layers, 2, 2)
        set_forward_backward_prefetch(model.gemma_expert.blocks, 2, 2)

        model: FSDPModule | PiPolicy

    global_batch_size = train_cfg.batch_size * train_cfg.grad_accum_steps * world_size
    lr = np.clip(train_cfg.lr * np.sqrt(global_batch_size), max=3e-4)
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
            optimizer=optimizer,
        )

    if global_rank == 0:
        run = wandb.init(
            project=train_cfg.wandb.project,
            mode=train_cfg.wandb.mode,
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
    last_time = time.perf_counter()
    for epoch in range(train_cfg.epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        log_tds = []

        pbar = range(len(dataloader) // train_cfg.grad_accum_steps)
        if global_rank == 0:
            pbar = tqdm(pbar, desc=f"Epoch {epoch+1}/{train_cfg.epochs}")

        model.train()
        data_loader_iter = iter(dataloader)
        for i in pbar:
            torch.cuda.nvtx.range_push("Zero Grad")
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

                torch.cuda.nvtx.range_push("Expand Data Sample")
                data_sample = data_sample.expand(
                    train_cfg.num_noise_per_sample, *data_sample.shape
                ).reshape(-1, *data_sample.shape[1:])
                prefix_pad_masks = prefix_pad_masks.expand(
                    train_cfg.num_noise_per_sample, *prefix_pad_masks.shape
                ).reshape(-1, *prefix_pad_masks.shape[1:])
                prefix_key_values = [
                    (
                        k.expand(train_cfg.num_noise_per_sample, *k.shape).reshape(
                            -1, *k.shape[1:]
                        ),
                        v.expand(train_cfg.num_noise_per_sample, *v.shape).reshape(
                            -1, *v.shape[1:]
                        ),
                    )
                    for k, v in prefix_key_values
                ]
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Noise Sampling")
                actions = data_sample.action_chunk.actions
                noise = sample_noise(actions.shape, device, dtype=actions.dtype)
                u_t = noise - actions
                timestep = sample_time(time_dist, data_sample.shape)
                noisy_actions = actions + timestep[:, None, None] * u_t
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Model Predict Suffix")
                v_t = model.predict_suffix(
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
                (loss / train_cfg.grad_accum_steps).backward()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Optimizer Step")
            norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg.clip_grad_norm
            )
            optimizer.step()
            torch.cuda.nvtx.range_pop()

            log_td = {}
            log_td["loss/flow_mse"] = loss.detach()
            if isinstance(norm_before_clip, torch.distributed.tensor.DTensor):
                norm_before_clip = norm_before_clip.full_tensor()
            log_td["loss/grad_norm"] = norm_before_clip
            log_td = TensorDict(log_td, [])
            log_td["loading"] = perf_dict.mean(dim=0)

            log_tds.append(log_td)

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
                    * train_cfg.log_interval
                    / elapsed_time
                )
                log_dict["perf/fps"] = fps
                log_dict["perf/fps.total"] = fps * world_size

                # log train stats
                log_td_stack: TensorDict = torch.stack(log_tds, dim=0)
                if world_size > 1:
                    log_td_stack.apply_(partial(dist.all_reduce, op=dist.ReduceOp.AVG))
                    # for tensor in log_td_stack.values(leaves_only=True):
                    #     dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                log_tds.clear()
                log_td_mean: TensorDict = log_td_stack.type(torch.float32).mean(dim=0)
                log_dict.update(
                    log_td_mean.flatten_keys(separator="/").to_dict(
                        convert_tensors=True
                    )
                )

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
                    # print(log_dict)
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

        save_checkpoint(model, optimizer, global_rank, f"checkpoint_{epoch+1}.pth")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
