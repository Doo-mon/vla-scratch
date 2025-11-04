from typing import Sequence

import torch
import numpy as np
from pathlib import Path

from vla_scratch.datasets.data_types import Observation
from vla_scratch.datasets.transforms import *
from vla_scratch.datasets.common import *
from vla_scratch.datasets.libero.common import *


class ToObservation(TransformFn):
    def compute(self, sample: Dict) -> Dict:
        observation = Observation(
            images=sample[PROCESSED_IMAGE_KEY],
            image_masks=sample[PROCESSED_IMAGE_MASK_KEY],
            state=sample[PROCESSED_STATE_KEY],
            tokenized_prompt=sample[TOKENIZED_KEY],
            tokenized_prompt_mask=sample[TOKENIZED_MASK_KEY],
        )
        return observation


class ToTorch(TransformFn):
    """Lightweight transform to convert incoming numpy state to torch tensor."""

    def compute(self, sample: Dict) -> Dict:
        result = {
            key: (
                torch.from_numpy(val).type(torch.float32)
                if isinstance(val, np.ndarray)
                else val
            )
            for key, val in sample.items()
        }
        return result


class ToNumpy(TransformFn):
    def compute(self, sample: Dict) -> Dict:
        numpy_sample = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                numpy_sample[key] = value.cpu().numpy()
            else:
                numpy_sample[key] = value
        return numpy_sample


class UnNormalize(TransformFn):
    """Percentile normalize state/actions and clip to [-1.5, 1.5].

    For each feature i, maps p2->-1 and p98->+1 via
        y_i = 2 * (x_i - p02_i) / (p98_i - p02_i) - 1
    then clips y to [-1.5, 1.5].

    The stats file may be .npz (numpy), .pt/.pth (torch.load dict), or .json.
    It should contain keys: "states_p02", "states_p98", "actions_p02", "actions_p98".
    Each array can be either length equal to the last-dimension size (per-feature),
    or equal to the total number of elements when the tensor is flattened
    (applied in flattened order then reshaped back).
    """

    def __init__(self, key, stats_file: Path, stats_key: str = None):
        stats_path = Path(stats_file)
        data = np.load(str(stats_path))

        def to_tensor(name: str) -> torch.Tensor:
            arr = data.get(name)
            return torch.from_numpy(arr).type(torch.float32)

        self.key = key
        if stats_key is None:
            stats_key = key
        self.p02 = to_tensor(f"{stats_key}_p02")
        self.p98 = to_tensor(f"{stats_key}_p98")

    def _unscale_clip(
        self, x: torch.Tensor, p02: torch.Tensor, p98: torch.Tensor
    ) -> torch.Tensor:
        x = x.clip(-1.5, 1.5)
        return unscale_transform(x, p02, p98)

    def compute(self, sample: Dict) -> Dict:
        actions = sample.get(self.key)
        sample[self.key] = self._unscale_clip(actions, self.p02, self.p98)
        return sample


if __name__ == "__main__":
    from vla_scratch.policies.pi.policy import PiPolicy
    from vla_scratch.datasets.data_types import DataSample

    observation_in = {
        "observation/image": np.random.randint(
            0, 255, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/wrist_image": np.random.randint(
            0, 255, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(9).astype(np.float32),
        "prompt": "Pick up the red block and place it on the green block.",
    }

    from transformers import PaliGemmaProcessor

    model_id = "google/paligemma-3b-mix-224"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    repo_root = Path(__file__).resolve().parents[2]
    input_transforms: Sequence[TransformFn] = [
        ConcatHistory(history=4),
        LiberoState(),
        Normalize(
            "state",
            stats_file=repo_root / "normalization_stats" / "libero_proprio_stats.npz",
            stats_key="states",
        ),
        StructurePrompt(),
        TokenizePrompt(tokenizer, max_length=128),
        PreprocessImage(),
        ToTensorClass(),
    ]
    observation = observation_in
    for transform in input_transforms:
        observation = transform.compute(observation)
    assert isinstance(observation, Observation)

    policy: PiPolicy = PiPolicy()

    with torch.inference_mode():
        actions = policy.sample_actions(observation, num_steps=10)
    print("Sampled actions shape:", actions.shape)

    state_orig = torch.from_numpy(observation[STATE_KEY]).type(torch.float32)
    output = {
        "actions": actions,
        "states": state_orig,  # the observation before transforms
    }

    output_transforms: Sequence[TransformFn] = [
        UnNormalize(
            key="actions", stats_file=Path("path/to/stats.npz"), stats_key="actions"
        ),
        LiberoActionToGlobal(),
        ToNumpy(),
    ]
    for transform in output_transforms:
        output = transform.compute(output)
