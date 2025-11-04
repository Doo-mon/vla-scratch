#!/usr/bin/env python3
import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import numpy as np
import torch

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf

from vla_scratch.datasets.common import PROCESSED_ACTION_KEY
from vla_scratch.datasets.config import (
    DataConfig,
    _instantiate_transform,
    create_dataset,
    _resolve_config_placeholders,
)
from vla_scratch.policies.config import PolicyConfig, create_policy

from vla_scratch.datasets.data_types import Observation
from vla_scratch.datasets.transforms import (
    TransformFn,
    Normalize,
    DeNormalize,
    load_norm_stats,
)
from vla_scratch.datasets.libero.transforms import (
    LiberoAction,
    LiberoActionToGlobal,
    CatHistory,
)
from vla_scratch.datasets.libero.common import STATE_KEY

from vla_scratch.serving.websocket_policy_server import WebsocketPolicyServer
from vla_scratch.serving.transforms import ToTorch, ToNumpy, ToObservation

logger = logging.getLogger(__name__)


@dataclass
class ServeConfig:
    defaults: list[Any] = field(
        default_factory=lambda: ["_self_", {"policy": "pi"}, {"data": "libero-ipec"}]
    )

    # server
    host: str = "0.0.0.0"
    port: int = 8000
    inference_steps: int = 10

    # configs
    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="serve", node=ServeConfig())


class ServePolicy:
    def __init__(
        self,
        model: torch.nn.Module,
        input_transforms: Sequence[TransformFn],
        output_transforms: Sequence[TransformFn],
        inference_steps: int = 10,
    ) -> None:
        self._model = model
        self._num_steps = inference_steps
        self._device = next(model.parameters()).device
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        observation = obs
        for transform in self._input_transforms:
            observation = transform.compute(observation)
        assert isinstance(observation, Observation)
        observation = observation.to(self._device).unsqueeze(0)

        actions = self._model.sample_actions(observation, num_steps=self._num_steps)

        state_orig = torch.from_numpy(obs[STATE_KEY]).type(torch.float32)
        output = {
            PROCESSED_ACTION_KEY: actions.squeeze(0).cpu(),
            STATE_KEY: state_orig,
        }
        for transform in self._output_transforms:
            output = transform.compute(output)
        return output

    def reset(self) -> None:
        for input_transform in self._input_transforms:
            if isinstance(input_transform, CatHistory):
                input_transform.buffer.clear()


class ReplayPolicy:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        input_transforms: Sequence[TransformFn],
        output_transforms: Sequence[TransformFn],
        inference_steps: int = 10,
    ) -> None:
        self._dataset = dataset
        self._counter = 0

        self._model = model
        self._num_steps = inference_steps
        self._device = next(model.parameters()).device
        self._num_steps = inference_steps
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        data_sample, _ = self._dataset[self._counter]
        actions = data_sample.action_chunk.actions.unsqueeze(0)
        self._counter += 1
        print(self._counter)

        state_orig = torch.from_numpy(obs[STATE_KEY]).type(torch.float32).unsqueeze(0)
        output = {
            PROCESSED_ACTION_KEY: actions.squeeze(0).cpu(),
            STATE_KEY: state_orig,
        }
        for transform in self._output_transforms:
            output = transform.compute(output)
        return output

    def reset(self) -> None:
        for input_transform in self._input_transforms:
            if isinstance(input_transform, CatHistory):
                input_transform.buffer.clear()


def _find_latest_checkpoint(
    path: Path, desired_iter: Optional[int] = None
) -> Optional[Path]:
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(path.glob("checkpoint_*.pth"))
    if not candidates:
        return None

    # pick the one with the largest numeric suffix if possible
    def _epoch_num(p: Path) -> int:
        stem = p.stem  # checkpoint_X
        try:
            epoch = int(stem.split("_")[-1])
            if desired_iter and epoch == desired_iter:
                epoch = 9999
            return epoch
        except Exception:
            return -1

    candidates_sorted = sorted(candidates, key=_epoch_num)
    return candidates_sorted[-1]


def _load_saved_cfgs(
    run_dir: Path,
) -> Tuple[Optional[DictConfig], Optional[DictConfig]]:
    policy_cfg_path = run_dir / "policy-cfg.yaml"
    data_cfg_path = run_dir / "data-cfg.yaml"
    policy_cfg = OmegaConf.load(policy_cfg_path) if policy_cfg_path.exists() else None
    data_cfg = OmegaConf.load(data_cfg_path) if data_cfg_path.exists() else None
    return policy_cfg, data_cfg


def _build_input_transforms(
    data_cfg: DataConfig, policy_cfg: PolicyConfig
) -> Sequence[TransformFn]:
    transforms: list[TransformFn] = []
    transforms.extend([CatHistory(history=policy_cfg.state_history), ToTorch()])

    # Instantiate dataset transforms, but skip LiberoAction (not available at serve time)
    dataset_transforms = [_instantiate_transform(spec) for spec in data_cfg.transforms]
    transforms.extend(
        [tf for tf in dataset_transforms if not isinstance(tf, LiberoAction)]
    )

    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        transforms.append(Normalize(norm_stats=stats))

    # Policy transforms (e.g., StructurePrompt, TokenizePrompt, PreprocessImage)
    policy_transforms = [_instantiate_transform(spec) for spec in policy_cfg.transforms]
    transforms.extend(policy_transforms)

    # Finally, convert standardized dict into Observation
    transforms.append(ToObservation())
    return transforms


def _build_output_transforms(
    data_cfg: DataConfig, policy_cfg: PolicyConfig
) -> Sequence[TransformFn]:
    transforms: list[TransformFn] = []

    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        transforms.append(DeNormalize(norm_stats=stats))

    transforms.extend([LiberoActionToGlobal(), ToNumpy()])
    return transforms


@hydra.main(config_name="serve", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # # Optionally override fields with saved configs from checkpoint dir
    # if cfg.get("checkpoint_path"):
    #     run_dir = Path(cast(str, cfg.checkpoint_path))
    #     saved_policy, saved_data = _load_saved_cfgs(run_dir)
    #     if saved_policy is not None:
    #         cfg.policy = OmegaConf.merge(cfg.policy, saved_policy)
    #     if saved_data is not None:
    #         cfg.data = OmegaConf.merge(cfg.data, saved_data)

    serve_cfg = cast(ServeConfig, OmegaConf.to_object(cfg))

    # Ensure data temporal parameters match policy
    serve_cfg.data.action_horizon = serve_cfg.policy.action_horizon
    serve_cfg.data.state_history = serve_cfg.policy.state_history

    dataset = create_dataset(
        serve_cfg.data,
        serve_cfg.policy,
    )

    dummy_data, _ = dataset[0]
    action_dim = dummy_data.action_chunk.actions.shape[-1]
    state_dim = dummy_data.observation.state.shape[-1]
    serve_cfg.policy.action_dim = action_dim
    serve_cfg.policy.state_dim = state_dim

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Create model from policy config
    print("Initializing model...")
    with torch.device(device):
        model = create_policy(serve_cfg.policy)
    print("Model initialized.")

    # Load latest checkpoint
    # serve_cfg.checkpoint_path = None
    if serve_cfg.checkpoint_path is not None:
        ckpt: Path = _find_latest_checkpoint(Path(serve_cfg.checkpoint_path))
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found under {serve_cfg.checkpoint_path}"
            )
        print(f"Loading checkpoint: {ckpt}")
        state_dict = torch.load(ckpt, map_location=device)
        model_state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        print("Checkpoint loaded.")
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    model.eval()

    # Build transforms
    input_transforms = _build_input_transforms(serve_cfg.data, serve_cfg.policy)
    output_transforms = _build_output_transforms(serve_cfg.data, serve_cfg.policy)

    # Wrap into serving policy
    policy = ServePolicy(
        model,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
        inference_steps=serve_cfg.inference_steps,
    )
    # policy = ReplayPolicy(
    #     dataset,
    #     model,
    #     input_transforms=input_transforms,
    #     output_transforms=output_transforms,
    #     inference_steps=serve_cfg.inference_steps,
    # )

    metadata = {
        "policy": serve_cfg.policy._target_.split(".")[-1],
        "device": str(device),
    }

    # Warmup once to trigger initialization
    warmup = True
    if warmup:
        observation_in = {
            "observation/image": np.random.randint(
                0, 255, size=(3, 480, 640), dtype=np.uint8
            ),
            "observation/wrist_image": np.random.randint(
                0, 255, size=(3, 480, 640), dtype=np.uint8
            ),
            "observation/state": np.random.rand(serve_cfg.policy.state_dim).astype(
                np.float32
            ),
            "task": "Pick up the red block and place it on the green block.",
        }
        policy.infer(observation_in)
        policy.reset()

    server = WebsocketPolicyServer(
        policy=policy, host=serve_cfg.host, port=serve_cfg.port, metadata=metadata
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(
        f"Serving policy {metadata.get('policy')} on {serve_cfg.host}:{serve_cfg.port} (host={hostname} ip={local_ip})",
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
