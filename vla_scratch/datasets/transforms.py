from typing import TypeAlias, Dict

import torch
import numpy as np
from pathlib import Path

from vla_scratch.datasets.data_types import ActionChunk, DataSample, Observation
from vla_scratch.datasets.math_utils import (
    matrix_from_quat,
    quat_mul,
    quat_conjugate,
    quat_from_angle_axis,
    quat_apply,
    quat_apply_inverse,
    unscale_transform,
    scale_transform,
)

DataDict: TypeAlias = Dict


class TransformFn:
    def compute(self, sample: DataDict) -> DataDict:
        raise NotImplementedError


def _rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrix must be (..., 3, 3), got {R.shape}")
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


class LiberoProprio(TransformFn):
    actions_low = torch.tensor([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, -1.0])
    actions_high = torch.tensor([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

    def compute(self, sample: DataDict) -> DataDict:
        """Transform LIBERO state/action into relative pose and target deltas.

        Expects sample to contain:
          - state: Tensor [T, D>=9] with layout [gripper0, gripper1, eef_pos(3), eef_quat_wxyz(4), ...]
                   where T = history + horizon and index of t is history
          - actions: Tensor [horizon, 7] with layout [dpos(3), daxis_angle(3), gripper(1)]

        Produces:
          - state: Tensor [H*(9 + 2)] = concat over history of [Δpos(3), Δori6d(6), gripper(2)]
          - actions: Tensor [horizon, 10] = per future step [Δpos_to_target(3), Δori6d_to_target(6), gripper(1)]
        """
        state_seq: torch.Tensor = sample["state"]
        actions_seq: torch.Tensor = sample["actions"]

        actions_seq = unscale_transform(
            actions_seq, LiberoProprio.actions_low, LiberoProprio.actions_high
        )

        T, state_dim = state_seq.shape
        horizon, action_dim = actions_seq.shape
        assert state_dim == 9, "Expected state dim 9 for LIBERO"
        assert action_dim == 7, "Expected action dim 7 for LIBERO"

        history = T - horizon
        if history < 0:
            raise ValueError(
                f"Invalid lengths: T={T}, K={horizon} imply history={history} < 0; expected T = H + 1 + K"
            )

        # Parse positions and orientations from state window
        pos_seq = state_seq[:, 2:5]
        quat_wxyz_seq = state_seq[:, 5:9]
        # Current frame index (t) in the loaded window
        current_t = history
        current_pos_w = pos_seq[current_t]
        current_quat_w = quat_wxyz_seq[current_t]

        # Build state: relative pose of past H states to current t, plus current grippers
        # Δpos in current frame
        history_pos_w = pos_seq[:history]  # [H, 3] corresponds to states t-H-1 ... t-2
        # Δpos: express in current frame using quaternion inverse apply
        current_quat_w_hist = current_quat_w.unsqueeze(0).expand(history, -1)
        history_dpos = quat_apply_inverse(
            current_quat_w_hist, history_pos_w - current_pos_w
        )
        # Δori: q_hist ⊖ q_current -> convert to 6D
        history_quat_w = quat_wxyz_seq[:history]
        history_dquat = quat_mul(quat_conjugate(current_quat_w_hist), history_quat_w)
        history_drotmat = matrix_from_quat(history_dquat)
        history_dori6d = _rotation_matrix_to_6d(history_drotmat)
        # gripper state
        history_grippers = state_seq[:history, 0:2]  # [H, 2] (not used in output state)

        state_vec = torch.cat(
            [history_dpos, history_dori6d, history_grippers], dim=-1
        )  # [H, 3 + 6 + 2]

        # Build future action targets (per-step target pose expressed relative to current t)
        future_pos_w = pos_seq[current_t : current_t + horizon]  # [K, 3]
        future_quat_w = quat_wxyz_seq[current_t : current_t + horizon]  # [K, 4]
        cmd_dpos = actions_seq[:, 0:3]
        cmd_drotvec = actions_seq[:, 3:6]
        cmd_grippers = actions_seq[:, 6:7]

        future_cmd_pos_w = future_pos_w + quat_apply(future_quat_w, cmd_dpos)

        delta_angle = torch.linalg.norm(cmd_drotvec, dim=-1)
        delta_quat = quat_from_angle_axis(delta_angle, cmd_drotvec)
        future_cmd_quat_w = quat_mul(future_quat_w, delta_quat)

        current_quat_w_expand = current_quat_w.unsqueeze(0).expand_as(future_cmd_quat_w)
        target_dpos = quat_apply_inverse(
            current_quat_w_expand, future_cmd_pos_w - current_pos_w
        )
        target_dquat = quat_mul(
            quat_conjugate(current_quat_w_expand), future_cmd_quat_w
        )
        target_drotmat = matrix_from_quat(target_dquat)
        target_dori6d = _rotation_matrix_to_6d(target_drotmat)

        actions_out = torch.cat([target_dpos, target_dori6d, cmd_grippers], dim=-1)

        sample["state"] = state_vec
        sample["actions"] = actions_out
        return sample


class ToTensorClass(TransformFn):
    def compute(self, sample: DataDict) -> DataDict:
        observation = Observation(
            images=sample["images"],
            image_masks=sample["image_masks"],
            state=sample["state"],
            tokenized_prompt=sample["tokenized_prompt"],
            tokenized_prompt_mask=sample["tokenized_prompt_mask"],
        )
        action = ActionChunk(actions=sample["actions"])
        data_sample = DataSample(observation=observation, action=action)
        return data_sample


class Normalize(TransformFn):
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

    def __init__(self, stats_file: Path):
        stats_path = Path(stats_file)
        data = np.load(str(stats_path))

        def to_tensor(name: str) -> torch.Tensor:
            arr = data.get(name)
            return torch.from_numpy(arr).type(torch.float32)

        self._state_p02 = to_tensor("states_p02")
        self._state_p98 = to_tensor("states_p98")
        self._action_p02 = to_tensor("actions_p02")
        self._action_p98 = to_tensor("actions_p98")

    def _scale_clip(
        self, x: torch.Tensor, p02: torch.Tensor, p98: torch.Tensor
    ) -> torch.Tensor:
        scaled = scale_transform(x, p02, p98)
        return torch.clamp(scaled, -1.5, 1.5)

    def compute(self, sample: DataDict) -> DataDict:
        state = sample.get("state")
        actions = sample.get("actions")
        sample["state"] = self._scale_clip(state, self._state_p02, self._state_p98)
        sample["actions"] = self._scale_clip(
            actions, self._action_p02, self._action_p98
        )
        return sample
