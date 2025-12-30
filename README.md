<div align="left">
  <h1>
  VLA-Scratch: 
  <!-- <br/> -->
  a Modular, Performant, Efficient Stack<br/>
  For Vision-Language-Action Models
  </h1>
</div>
<!-- add a huggingface badge, a twitter badge and a github star badge -->

## ✨ Key-Features

- **Explicit Data Model for Composable Modules** 
    - `TensorClass` is used to define explicit data boundaries between modules, making our codebase fully typed and modular. This allows heterogeneous dataset co-training, and [clearer data flow](vla_scratch/transforms/README.md) between datasets, policies, and transforms.
    ![data model](assets/data_model.png)
- **Dedicated Tuning for a First-Class Performance Stack** 
    - The [Qwen3-VL bridge](vla_scratch/policies/modules/vlm_bridge/qwen/bridge.py) rewrites the forward path to [minimize host-device syncs](vla_scratch/policies/README.md), making throughput 2x higher.
    - Layer-wise FSDP sharding and gradient checkpointing saves memory up to 2x, making it easier to scale model parameters.
    - TODO: figure here.
- **Clarity-Focused Hydra Workflow for Seamless Experimentation** 
    - Every [policy](vla_scratch/policies/pi/config.py) and [data](vla_scratch/datasets/config.py) config is registered with Hydra's `ConfigStore`, so experiments are overrideable with minimal boilerplate.
    - Training, eval, and serving scripts share a common config grammar, so switching between workflows is seamless.
    - TODO: example script snippet here.


## 🎯 Capabilities

TODO: Heterogeneous dataset co-training (VQA, Action), Multi VLM backbone, Simulation-ready Serving scripts, feature rich visualizations, etc.

## 🗂️ Codebase Structure

VLA-Scratch is a fully modular, high-performance VLA stack built around Hydra configs, TensorClass data models, and reusable training helpers.

| Path | Description | Docs |
| --- | --- | --- |
| `vla_scratch/transforms/` | Data transforms and TensorClass models | [Documentation →](vla_scratch/transforms/README.md) |
| `vla_scratch/datasets/` | Dataset loaders and transforms | [Documentation →](vla_scratch/datasets/README.md) |
| `vla_scratch/policies/` | Policy interfaces, bridges, and action heads | [Documentation →](vla_scratch/policies/README.md) |
| `scripts/` | Training/eval/serving scripts with shared Hydra grammar | [Documentation →](scripts/README.md) |
<!-- - `vla_scratch/helpers/` — repo-aware data + training helpers (prefetch dataloaders, FSDP setup, normalization I/O).
- `vla_scratch/utils/` — standalone utilities (config discovery, checkpointing, filesystem paths) that avoid repo-level imports.
- `examples/` — ready-to-run experiment folders demonstrating LIBERO usage and viz workflows.
- `tests/` — smoke tests and unit tests for policies, transforms, and helpers. -->

## 🚀 Quickstart

```bash
# Setup virtual environment and install dependencies:
uv sync
source .venv/bin/activate

# For torch compile and lerobot decode video
apt update
apt install -y python3.10-dev ffmpeg 
```

### Training
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    train_policy.py \
    policy=pi-qwen \
    policy.state_history=0 \
    policy.action_horizon=10 \
    policy.use_state=False \
    policy.transforms.0.max_length=500 \
    data=libero-spatial \
    batch_size=32 \
    eval_data=libero-spatial \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    wandb.mode=online
```

<!-- 

```bash
hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset

export HF_CACHE_TEMP=/home/elijah/.cache/huggingface/
ln -s ${HF_CACHE_TEMP}/hub/datasets--IPEC-COMMUNITY--libero_spatial_no_noops_1.0.0_lerobot/snapshots/cb7508999d4a8caa65b6448d6d700e1f347b809e ${HF_CACHE_TEMP}/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
```
 -->
