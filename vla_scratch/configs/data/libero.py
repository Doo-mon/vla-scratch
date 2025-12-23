from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.libero.config import libero_spatial_config

cs = ConfigStore.instance()

libero_spatial_eval_cfg = EvalDataCfg(
    datasets={
        "spatial": EvalDatasetCfg(
            data=libero_spatial_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        )
    }
)
cs.store(name="libero_spatial", node=libero_spatial_eval_cfg, group="eval_data")
