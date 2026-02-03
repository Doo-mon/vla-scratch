## Training
```bash
# LIBERO with Qwen3-VL
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_policy.py \
    policy=pi-qwen \
    policy.state_history=0 \
    policy.action_horizon=30 \
    policy.use_state=False \
    policy.transforms.0.max_length=180 \
    data=libero-spatial \
    eval_data=libero-spatial \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    wandb.mode=online

# LIBERO with PaliGemma
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_policy.py \
    policy=pi-paligemma \
    policy.state_history=0 \
    policy.action_horizon=30 \
    policy.use_state=False \
    policy.transforms.0.max_length=550 \
    data=libero-spatial \
    eval_data=libero-spatial \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    wandb.mode=online
```

### Upload Checkpoints

The training output structure is:

```
outputs/<YYYY-MM-DD>/<HH-MM-SS>-<exp_name>/
├── cfg.yaml                    # training config snapshot
└── checkpoint_<epoch>/         # checkpoint directory per save interval
    ├── model.pt                # model weights
    └── optimizer.pt            # optimizer state (excluded from upload)
```

Use the command below to upload the latest checkpoint (automatically picks the highest-numbered `checkpoint_*`) and `cfg.yaml` to Hugging Face:

```bash
uv run bash scripts/helpers/upload_checkpoint.sh <user_name/repo_name> <checkpoint_path>
```

## Evaluation
See [examples](examples/README.md) for details about evaluation in LIBERO and other simulation environments.
