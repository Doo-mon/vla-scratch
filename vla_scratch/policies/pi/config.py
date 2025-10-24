import dataclasses
from vla_scratch.policies.modules.dit import Variant


@dataclasses.dataclass
class PiConfig:
    action_expert_variant: Variant = "300m"
    action_dim: int = 32
    action_horizon: int = 30
    state_dim: int = 24
    state_history: int = 10
