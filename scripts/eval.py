#!/usr/bin/env python3
from __future__ import annotations

"""
Simple evaluation script: load Moz dataset + Pi policy, run sampling inference,
and compute MSE loss between predicted actions and dataset target actions.

References:
- Dataset/policy loading adapted from scripts/vis/pub_policy_eval_server.py
- MSE evaluation mirrors train_policy.compute_sample_mse
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from vla_scratch.datasets.spirit.config import MozConfig
from vla_scratch.datasets.config import create_dataset, _instantiate_transform
from vla_scratch.policies.pi.config import PiConfig
from vla_scratch.policies.config import create_policy

from vla_scratch.datasets.data_types import DataSample


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate policy MSE on Moz dataset")
    # Dataset args
    p.add_argument("--repo-id", type=str, default="20251013_sdandardpp4of12_pickplace_ori")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="Root directory containing dataset repo folder",
    )
    p.add_argument("--episodes", type=str, default=None, help="Comma/range list, e.g., 0,1,2 or 0-3 (optional)")
    # Eval controls
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=512, help="Number of samples to evaluate (max)")
    p.add_argument("--num-steps", type=int, default=10, help="Sampling steps for model.sample_actions")
    # Policy
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p


def _parse_episodes(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None or spec.strip() == "":
        return None
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    out: List[int] = []
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return out


def _find_latest_checkpoint(path: Path) -> Optional[Path]:
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(path.glob("checkpoint_*.pth"))
    if not candidates:
        return None
    return candidates[-1]


@torch.inference_mode()
def compute_mse(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_sample_steps: int,
) -> float:
    errs: List[torch.Tensor] = []
    it = iter(dataloader)
    from tqdm import tqdm
    for _ in tqdm(range(len(dataloader))):
        batch, _ = next(it)
        batch: DataSample = batch.to(device)
        pred = model.sample_actions(batch.observation, num_steps=num_sample_steps)
        target = batch.action_chunk.actions
        se = F.mse_loss(pred, target, reduction="none").mean()
        errs.append(se)
    return float(torch.stack(errs).mean().item())


def main() -> None:
    args = build_argparser().parse_args()

    # Data + policy configs
    data_cfg = MozConfig()
    data_cfg.repo_id = args.repo_id
    data_cfg.root_path = args.root
    data_cfg.episodes = list(range(2))
    # _parse_episodes(args.episodes)

    policy_cfg = PiConfig()

    policy_cfg.action_horizon = 30
    policy_cfg.state_history = 1
    policy_cfg.num_obs_registers = 4
    policy_cfg.expert_only_use_register = True

    # Keep temporal params in sync
    data_cfg.action_horizon = policy_cfg.action_horizon
    data_cfg.state_history = policy_cfg.state_history

    # Disable image augmentation for eval
    for i, spec in enumerate(list(data_cfg.transforms)):
        if isinstance(spec, dict) and spec.get("_target_") == "vla_scratch.datasets.spirit.transforms.SpiritImages":
            spec.update({"enable_aug": False, "aug_p": 0.0})
            data_cfg.transforms[i] = spec

    # Create transformed dataset (includes normalization + policy transforms + ToTensorClass)
    dataset = create_dataset(data_cfg, policy_cfg)

    # Infer dims from one sample
    sample0, _ = dataset[0]
    policy_cfg.state_dim = int(sample0.observation.state.shape[-1])
    policy_cfg.action_dim = int(sample0.action_chunk.actions.shape[-1])

    # Model
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.device(device):
        model = create_policy(policy_cfg)
    if args.checkpoint is not None:
        ckpt = _find_latest_checkpoint(Path(args.checkpoint))
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found under {args.checkpoint}")
        state_dict = torch.load(ckpt, map_location=device)
        print(f"[eval] loading model checkpoint from {ckpt}")
        model_state_dict = state_dict.get("model", state_dict)
        # print missing/unexpected keys
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        if missing:
            print(f"[eval] warning: missing keys in model state_dict: {missing}")
        if unexpected:
            print(f"[eval] warning: unexpected keys in model state_dict: {unexpected}")
    model.eval()

    # Dataloader — subset for speed
    total = len(dataset)
    num = min(int(args.num_samples), total)
    indices = list(range(num))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )

    # Evaluate MSE
    mse = compute_mse(model, loader, device, num_sample_steps=int(args.num_steps))
    print(f"Eval MSE over {num} samples (batch={args.batch_size}, steps={args.num_steps}): {mse:.6f}")


if __name__ == "__main__":
    main()

