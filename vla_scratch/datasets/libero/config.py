from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig


@dataclass
class LiberoIPECConfig(DataConfig):
    @staticmethod
    def _default_input_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoState"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]

    @staticmethod
    def _default_output_transform_configs() -> list[Dict[str, Any]]:
        # Training target assembly
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoAction"},
        ]

    @staticmethod
    def _default_output_inv_transform_configs() -> list[Dict[str, Any]]:
        # Map predicted actions back to dataset/world frame
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToGlobal"},
        ]

    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: List[str] = field(
        default_factory=lambda: [
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
        ]
    )
    input_transforms: List[Any] = field(
        default_factory=_default_input_transform_configs
    )
    output_transforms: List[Any] = field(
        default_factory=_default_output_transform_configs
    )
    output_inv_transforms: List[Any] = field(
        default_factory=_default_output_inv_transform_configs
    )

    norm_stats_path: str = (
        "normalization_stats/libero/IPEC-COMMUNITY/libero_no_noops_1.0.0_lerobot-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    )


@dataclass
class LiberoIPECGlobalConfig(DataConfig):
    @staticmethod
    def _default_input_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoGlobalState"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]

    @staticmethod
    def _default_output_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoAction"},
        ]

    @staticmethod
    def _default_output_inv_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToGlobal"},
        ]

    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: List[str] = field(
        default_factory=lambda: [
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
        ]
    )
    input_transforms: List[Any] = field(
        default_factory=_default_input_transform_configs
    )
    output_transforms: List[Any] = field(
        default_factory=_default_output_transform_configs
    )
    output_inv_transforms: List[Any] = field(
        default_factory=_default_output_inv_transform_configs
    )

    norm_stats_path: str = (
        "normalization_stats/libero/IPEC-COMMUNITY/libero_no_noops_1.0.0_lerobot-global-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    )


@dataclass
class LiberoIPECDummyConfig(DataConfig):
    @staticmethod
    def _default_input_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoState"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]

    @staticmethod
    def _default_output_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionDummy"},
        ]

    @staticmethod
    def _default_output_inv_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToGlobal"},
        ]

    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: str = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
    input_transforms: List[Any] = field(
        default_factory=_default_input_transform_configs
    )
    output_transforms: List[Any] = field(
        default_factory=_default_output_transform_configs
    )
    output_inv_transforms: List[Any] = field(
        default_factory=_default_output_inv_transform_configs
    )

    norm_stats_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="libero-ipec", node=LiberoIPECConfig(), group="data")
cs.store(name="libero-ipec-global", node=LiberoIPECGlobalConfig(), group="data")
cs.store(
    name="libero-ipec-spatial",
    node=LiberoIPECGlobalConfig(
        repo_id=["IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"]
    ),
    group="data",
)
cs.store(name="libero-ipec-dummy", node=LiberoIPECDummyConfig(), group="data")
