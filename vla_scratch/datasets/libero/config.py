from dataclasses import dataclass

from vla_scratch.datasets.config import DataConfig
from hydra.core.config_store import ConfigStore


@dataclass
class LiberoConfig(DataConfig):
    _target_: str = "vla_scratch.datasets.libero.dataset.LIBERODataset"
    repo_id: str = "elijahgalahad/libero_spatial_noops_v30"
    norm_stats_path: str = (
        "normalization_stats/libero/lerobot-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    )


libero_spatial_config = LiberoConfig()

cs = ConfigStore.instance()
cs.store(name="libero-spatial", node=libero_spatial_config, group="data")
