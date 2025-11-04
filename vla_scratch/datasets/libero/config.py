from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vla_scratch.datasets.config import DataConfig


@dataclass
class LiberoIPECConfig(DataConfig):
    @staticmethod
    def _default_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoState"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoAction"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]

    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: List[str] = field(
        default_factory=lambda: [
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
        ]
    )

    transforms: List[Any] = field(default_factory=_default_transform_configs)

    # norm_stats_path: Optional[Path] = None
    norm_stats_path: Optional[str] = (
        "normalization_stats/libero/IPEC-COMMUNITY/libero_no_noops_1.0.0_lerobot-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    )


@dataclass
class LiberoIPECSpatialConfig(DataConfig):
    @staticmethod
    def _default_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoState"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoAction"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]

    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: List[str] = field(
        default_factory=lambda: [
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
        ]
    )

    transforms: List[Any] = field(default_factory=_default_transform_configs)

    # norm_stats_path: Optional[Path] = None
    norm_stats_path: Optional[str] = (
        "normalization_stats/libero/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot.npz"
    )


@dataclass
class LiberoIPECDummyConfig(DataConfig):
    @staticmethod
    def _default_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoState"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionDummy"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]

    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: str = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"

    transforms: List[Any] = field(default_factory=_default_transform_configs)

    norm_stats_path: Optional[Path] = None


from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="libero-ipec", node=LiberoIPECConfig, group="data")
cs.store(name="libero-ipec-spatial", node=LiberoIPECSpatialConfig, group="data")
cs.store(name="libero-ipec-dummy", node=LiberoIPECDummyConfig, group="data")
