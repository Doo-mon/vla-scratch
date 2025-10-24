import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import h5py
import torch
import torch.nn.functional as F

from vla_scratch.datasets.base import Dataset as BaseDataset


@dataclass(frozen=True)
class _IndexEntry:
    file_idx: int
    episode_key: str
    step_index: int


@dataclass(frozen=True)
class _DatasetInfo:
    path: Path
    prompt: str
    tokenised_prompt: torch.Tensor
    tokenised_prompt_mask: torch.Tensor


libero_data_cfg = {
    "state": [
        {
            "name": "gripper_0",
            "type": "scaler",
        },
        {
            "name": "gripper_1",
            "type": "scaler",
        },
        {
            "name": "eef_pos",
            "type": "pos",
        },
        {
            "name": "eef_quat",
            "type": "quat_wxyz",
        },
    ],
    "actions": [
        {
            "name": "delta_eef_pos",
            "type": "pos",
        },
        {
            "name": "delta_eef_axis_angle",
            "type": "axis_angle",
        },
        {
            "name": "gripper_action",
            "type": "scaler",
        },
    ],
}


class LiberoDataset(BaseDataset):
    """PyTorch dataset for LIBERO demonstrations."""

    def __init__(
        self,
        root: str | Path,
        tokenizer,
        cameras: Sequence[str] | None = None,
        *,
        action_horizon: int = 1,
        state_history: int = 1,
        image_dtype: torch.dtype = torch.float32,
        max_tokens: int = 256,
        profile: bool = False,
    ) -> None:
        super().__init__()
        self._src_h5_paths = self._resolve_sources(root)
        self._cameras = tuple(cameras) if cameras is not None else None
        self._tokenizer = tokenizer
        self._image_dtype = image_dtype
        self._max_tokens = max_tokens
        self._state_history = int(state_history)
        self._action_horizon = int(action_horizon)
        self._target_image_size = (224, 224)

        self._datasets: list[_DatasetInfo] = []
        self._indices: list[_IndexEntry] = []
        self._action_dim: int | None = None
        self._state_dim: int | None = None
        self._processed_image_shape: tuple[int, int, int] | None = None
        self._num_cameras: int = 0

        self._profile_enabled = bool(profile)
        self._profile_totals: dict[tuple[str, ...], float] = {}
        self._profile_counts: dict[tuple[str, ...], int] = {}
        self._profile_stack: list[tuple[tuple[str, ...], float]] = []

        self._build_index()
        if not self._indices:
            raise RuntimeError("No samples found in LIBERO dataset.")
        self._inspect_structure()

        self._emit_profile(len(self._indices))

    def _resolve_sources(self, root: str | Path) -> list[Path]:
        path = Path(root)
        if not path.exists():
            raise FileNotFoundError(f"LIBERO dataset path not found: {path}")

        if path.is_file():
            return [path]

        candidates = sorted(
            {
                *path.rglob("*.hdf5"),
                *path.rglob("*.h5"),
            }
        )
        if not candidates:
            raise FileNotFoundError(f"No HDF5 files discovered under {path}")
        return candidates

    def _build_index(self) -> None:
        self._profile_push_event("build_index")
        flattened: list[_IndexEntry] = []
        for file_idx, dataset_path in enumerate(self._src_h5_paths):
            info, entries = self._index_single_file(
                dataset_path,
                file_idx,
            )
            self._datasets.append(info)
            flattened.extend(entries)
        self._indices = flattened
        self._profile_pop_event()

    def _inspect_structure(self) -> None:
        reference_entry = self._indices[0]
        dataset_info = self._datasets[reference_entry.file_idx]

        with h5py.File(dataset_info.path, "r") as handle:
            data_group = handle["data"]
            episode_group = data_group[reference_entry.episode_key]
            obs_group = episode_group["obs"]

            if self._cameras is None:
                cameras = [key for key in obs_group.keys() if key.endswith("_rgb")]
                if not cameras:
                    cameras = list(obs_group.keys())
                cameras.sort()
                self._cameras = tuple(cameras)
            else:
                missing = [
                    camera for camera in self._cameras if camera not in obs_group
                ]
                if missing:
                    raise KeyError(
                        f"Requested cameras {missing} not found in episode {reference_entry.episode_key}"
                    )

            if not self._cameras:
                raise ValueError("No camera streams available in episode")

            sample_camera = obs_group[self._cameras[0]]
            if sample_camera.ndim != 4:
                raise ValueError("Expected camera dataset to have shape (T, H, W, C)")
            _, height, width, channels = sample_camera.shape
            self._processed_image_shape = (int(channels), *self._target_image_size)
            self._num_cameras = len(self._cameras)

            state_dataset = episode_group["robot_states"]
            if state_dataset.ndim < 2:
                raise ValueError("Robot states dataset must be at least 2D")
            self._state_dim = int(state_dataset.shape[-1])

            action_dataset = episode_group["actions"]
            if action_dataset.ndim < 2:
                raise ValueError("Actions dataset must be at least 2D")
            self._action_dim = int(action_dataset.shape[-1])

    def _tokenize_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        call_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self._max_tokens,
            "return_tensors": "pt",
        }

        forward = getattr(self._tokenizer, "forward", None)
        if callable(forward):
            encoded = forward(prompt, **call_kwargs)
        else:
            callable_tokenizer = getattr(self._tokenizer, "__call__", None)
            if callable(callable_tokenizer):
                encoded = callable_tokenizer(prompt, **call_kwargs)
            else:
                tokens = self._tokenizer.encode(prompt)
                if len(tokens) > self._max_tokens:
                    tokens = tokens[: self._max_tokens]
                pad_len = self._max_tokens - len(tokens)
                mask = [True] * len(tokens) + [False] * pad_len
                tokens = tokens + [0] * pad_len
                token_tensor = torch.tensor(tokens, dtype=torch.long)
                mask_tensor = torch.tensor(mask, dtype=torch.bool)
                return token_tensor, mask_tensor

        tokens = encoded["input_ids"][0].to(dtype=torch.long)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            mask = attention_mask[0].to(dtype=torch.bool)
        return tokens, mask

    def _load_images(self, episode_group: h5py.Group, step_index: int) -> torch.Tensor:
        assert self._cameras is not None
        images: list[torch.Tensor] = []
        obs_group = episode_group["obs"]
        for camera in self._cameras:
            frame_np = np.asarray(obs_group[camera][step_index], dtype=np.uint8)
            frame = (
                torch.from_numpy(frame_np)
                .permute(2, 0, 1)
                .contiguous()
                .to(torch.float32)
            )
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=self._target_image_size,
                mode="bilinear",
                align_corners=False,
            )[0]
            frame = frame.div_(255.0)
            frame = frame.sub_(0.5).div_(0.5)
            frame = frame.flip(dims=[1]).contiguous()
            images.append(frame.to(dtype=self._image_dtype))
        return torch.stack(images, dim=0)

    def _load_state(self, episode_group: h5py.Group, step_index: int) -> torch.Tensor:
        history_start = step_index - self._state_history
        history_end = step_index + self._action_horizon
        state = np.asarray(
            episode_group["robot_states"][history_start:history_end], dtype=np.float32
        )
        return torch.from_numpy(state)

    def _load_actions(self, episode_group: h5py.Group, step_index: int) -> torch.Tensor:
        horizon_stop = step_index + self._action_horizon
        raw_actions = np.asarray(
            episode_group["actions"][step_index:horizon_stop], dtype=np.float32
        )
        return torch.from_numpy(raw_actions)

    def __len__(self) -> int:  # noqa: D401 - inherited behaviour
        return len(self._indices)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self._indices):
            raise IndexError(index)
        entry = self._indices[index]
        dataset_info = self._datasets[entry.file_idx]

        with h5py.File(dataset_info.path, "r") as handle:
            episode_group = handle["data"][entry.episode_key]
            images = self._load_images(episode_group, entry.step_index)
            image_masks = torch.ones(self._num_cameras, 1, dtype=torch.bool)
            state = self._load_state(episode_group, entry.step_index)
            actions = self._load_actions(episode_group, entry.step_index)

        data_dict = dict(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=dataset_info.tokenised_prompt,
            tokenized_prompt_mask=dataset_info.tokenised_prompt_mask,
            actions=actions,
        )
        return data_dict

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("Action dimension metadata is unavailable")
        return self._action_dim

    @property
    def state_dim(self) -> int:
        if self._state_dim is None:
            raise RuntimeError("State dimension metadata is unavailable")
        return self._state_dim

    @property
    def image_shape(self) -> tuple[int, int, int]:
        if self._processed_image_shape is None:
            raise RuntimeError("Image shape metadata is unavailable")
        return self._processed_image_shape

    def episode_keys(self) -> Iterable[str]:
        """Return the episode identifiers present in the dataset."""
        return tuple(
            f"{self._datasets[index.file_idx].path.name}:{index.episode_key}"
            for index in self._indices
        )

    def _profile_push_event(self, name: str) -> None:
        if not self._profile_enabled:
            return
        parent_path = self._profile_stack[-1][0] if self._profile_stack else ()
        path = (*parent_path, name)
        self._profile_stack.append((path, time.perf_counter()))

    def _profile_pop_event(self) -> None:
        if not self._profile_enabled:
            return
        if not self._profile_stack:
            return
        path, start = self._profile_stack.pop()
        duration = time.perf_counter() - start
        self._profile_totals[path] = self._profile_totals.get(path, 0.0) + duration
        self._profile_counts[path] = self._profile_counts.get(path, 0) + 1

    def _emit_profile(self, num_samples: int) -> None:
        if not self._profile_enabled or not self._profile_totals:
            return
        if self._profile_stack:
            self._profile_stack.clear()
        print("[LiberoDataset] Profiling summary")
        print(f"  samples: {num_samples}")
        tree: dict[str, dict[str, Any]] = {}
        for path, total in self._profile_totals.items():
            count = self._profile_counts.get(path, 0)
            node = tree
            for idx, name in enumerate(path):
                entry = node.setdefault(
                    name, {"stats": {"total": 0.0, "count": 0}, "children": {}}
                )
                if idx == len(path) - 1:
                    entry["stats"]["total"] = total
                    entry["stats"]["count"] = count
                node = entry["children"]

        rows: list[tuple[int, tuple[str, ...], float, int]] = []

        def collect(
            node_dict: dict[str, dict[str, Any]],
            path_prefix: tuple[str, ...],
            depth: int,
        ) -> None:
            items = sorted(
                node_dict.items(),
                key=lambda item: item[1]["stats"]["total"],
                reverse=True,
            )
            for name, content in items:
                stats: dict[str, Any] = content["stats"]
                children: dict[str, dict[str, Any]] = content["children"]
                total = stats["total"]
                count = stats["count"]
                label_path = path_prefix + (name,)
                rows.append((depth, label_path, total, count))
                collect(children, label_path, depth + 1)

        collect(tree, tuple(), 1)

        if not rows:
            return

        labels = [f"{'/'.join(path)}:" for depth, path, _, _ in rows]
        label_width = max(len(label) for label in labels)
        totals = [f"{total:.3f}s total" for _, _, total, _ in rows]
        total_width = max(len(total_str) for total_str in totals)
        ms_values = []
        for _, _, total, count in rows:
            per_call_ms = (total / count) * 1e3 if count else 0.0
            ms_values.append(f"{per_call_ms:.3f}ms/call")
        ms_width = max(len(ms) for ms in ms_values)

        for label, total_str, ms_str in zip(labels, totals, ms_values):
            print(
                f"{label:<{label_width}} {total_str:>{total_width}} {ms_str:>{ms_width}}"
            )

    def _index_single_file(
        self,
        dataset_path: Path,
        file_idx: int,
    ) -> tuple[_DatasetInfo, list[_IndexEntry]]:
        with h5py.File(dataset_path, "r") as handle:
            if "data" not in handle:
                raise KeyError(
                    f"Expected group 'data' in LIBERO dataset: {dataset_path}"
                )
            data_group = handle["data"]

            try:
                problem_info = data_group.attrs.get("problem_info")
                info = json.loads(problem_info)
                prompt = info.get("language_instruction", "") or ""
                prompt = prompt.strip().replace("_", " ").replace("\n", " ")
                prompt = f"<bos>Task: {prompt};\n Action:"
            except Exception:  # noqa: BLE001 - best effort parsing
                prompt = ""

            tokens, mask = self._tokenize_prompt(prompt)
            dataset_info = _DatasetInfo(
                path=dataset_path,
                prompt=prompt,
                tokenised_prompt=tokens,
                tokenised_prompt_mask=mask,
            )

            entries: list[_IndexEntry] = []
            for episode_key in sorted(data_group.keys()):
                episode_group = data_group[episode_key]
                num_samples = episode_group.attrs.get("num_samples").item()

                max_start = num_samples - self._action_horizon
                min_start = self._state_history
                if max_start < 0:
                    continue
                if min_start > max_start:
                    continue

                for step in range(min_start, max_start + 1):
                    entries.append(_IndexEntry(file_idx, episode_key, step))

            return dataset_info, entries


if __name__ == "__main__":
    from vla_scratch.datasets.base import TransformedDataset
    from vla_scratch.datasets.transforms import LiberoProprio, ToTensorClass, Normalize
    from vla_scratch.datasets.data_types import DataSample

    class _ScriptTokenizer:
        def encode(self, prompt: str):  # noqa: D401 - minimal stub
            encoded = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8)
            return encoded.tolist()

        def decode(self, ids):
            if isinstance(ids, (list, tuple)):
                buffer = bytes(int(x) for x in ids if int(x) != 0)
            elif isinstance(ids, np.ndarray):
                buffer = bytes(int(x) for x in ids.tolist() if int(x) != 0)
            else:
                buffer = bytes(ids)
            return buffer.decode("utf-8", errors="ignore")

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "datasets" / "libero_spatial"
    tokenizer = _ScriptTokenizer()
    dataset = LiberoDataset(
        dataset_dir,
        tokenizer=tokenizer,
        action_horizon=30,
        state_history=10,
    )
    transforms = [LiberoProprio(), Normalize(stats_file=repo_root / "normalization_stats" / "libero_proprio_stats.npz"), ToTensorClass()]
    dataset = TransformedDataset(dataset, transforms)

    data_sample: DataSample = dataset[0]
    observation, actions = data_sample.observation, data_sample.action.actions

    print("Loaded sample from:", dataset._dataset.episode_keys()[0])
    for key, img in zip(dataset._dataset._cameras, observation.images):
        print(f"  {key}:", tuple(img.shape), img.dtype)

    print("ids:", observation.tokenized_prompt.tolist())
    prompt = tokenizer.decode(observation.tokenized_prompt.tolist())
    print("Prompt:", prompt)

    print("State shape:", tuple(observation.state.shape))
    if actions is not None:
        print("Action shape:", tuple(actions.shape))

    # import matplotlib and use histogram to visualize the distribution of each action dimension
    # first create a dataloader to load the dataset in batches
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        collate_fn=torch.stack,
    )

    states = []
    actions = []
    for i in tqdm(range(32)):
        data_sample = next(iter(dataloader))
        states.append(data_sample.observation.state)
        actions.append(data_sample.action.actions)
        del data_sample
    
    states = torch.cat(states, dim=0).numpy()[:, ::5, :]
    actions = torch.cat(actions, dim=0).numpy()[:, ::10, :]
    states = states.reshape(states.shape[0], -1)
    actions = actions.reshape(actions.shape[0], -1)
    
    # visualize histogram
    import matplotlib.pyplot as plt
    num_cols = 5

    num_state_dims = states.shape[1]
    num_state_rows = (num_state_dims + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_state_rows, num_cols, figsize=(4 * num_cols, 3 * num_state_rows))
    for i in range(num_state_dims):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].hist(states[:, i], bins=50, color='blue', alpha=0.7)
        axes[row, col].set_title(f'State Dimension {i}')
    plt.tight_layout()

    num_action_dims = actions.shape[1]
    num_action_rows = (num_action_dims + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_action_rows, num_cols, figsize=(4 * num_cols, 3 * num_action_rows))
    for i in range(num_action_dims):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].hist(actions[:, i], bins=50, color='green', alpha=0.7)
        axes[row, col].set_title(f'Action Dimension {i}')
    plt.tight_layout()
    plt.show()
    
    breakpoint()
