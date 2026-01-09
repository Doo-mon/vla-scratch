### Training
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_policy.py \
    policy=pi-qwen \
    policy.state_history=1 \
    policy.action_horizon=30 \
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

### Evaluation
See [examples](examples/README.md) for details about evaluation in LIBERO and other simulation environments.