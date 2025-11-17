from dataclasses import dataclass, field
from typing import Optional, Any, List
from hydra.core.config_store import ConfigStore
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.policies.modules.dit import DiTConfig


def _default_pi_transforms() -> list[Any]:
    return [
        {
            "_target_": "vla_scratch.policies.pi.transforms.PreprocessImage",
            "target_size": (224, 224),
        }
    ]


@dataclass
class PiConfig(PolicyConfig):
    _target_: str = "vla_scratch.policies.pi.policy.PiPolicy"

    transforms: List[Any] = field(default_factory=_default_pi_transforms)

    action_expert_cfg: DiTConfig = DiTConfig(
        hidden_size=1024,
        num_hidden_layers=12,
        intermediate_size=4096,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
    )
    model_id: str = "google/paligemma-3b-mix-224"
    # Name of the VLM class from `transformers` to instantiate.
    # Examples: "PaliGemmaForConditionalGeneration", "Qwen3VLForConditionalGeneration"
    vlm_type: str = "PaliGemmaForConditionalGeneration"
    # Max text tokens for VLM tokenization (wrapper-specific)
    max_prompt_length: int = 64

    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    state_history: int = 10
    action_horizon: int = 30

    use_state: bool = True

    num_obs_registers: int = 0
    expert_only_use_register: bool = False


Cs = ConfigStore.instance()
Cs.store(name="pi", node=PiConfig(), group="policy")
Cs.store(
    name="pi-qwen",
    node=PiConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        vlm_type="Qwen3VLForConditionalGeneration",
        max_prompt_length=256,
        action_expert_cfg= DiTConfig(
            hidden_size=1024,
            num_hidden_layers=12,
            intermediate_size=4096,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=256,)
    ),
    group="policy",
)
