# LIBERO Eval Example

Simulator env setup (`examples/libero/.venv`):
```bash
git clone git@github.com:Lifelong-Robot-Learning/LIBERO.git ../LIBERO
export LIBERO_ROOT=$(pwd)/../LIBERO

uv sync --project examples/libero 
source examples/libero/.venv/bin/activate
uv pip install -r $LIBERO_ROOT/requirements.txt
uv pip install -e $LIBERO_ROOT
export PYTHONPATH=$PYTHONPATH:$LIBERO_ROOT
```

Start up policy server:
```bash
source .venv/bin/activate
python scripts/serve_policy.py \
  policy=pi-qwen \
  policy.state_history=1 \
  policy.action_horizon=20 \
  data=libero-ipec-spatial \
  checkpoint_path=hf:elijahgalahad/libero_policy
```

Start sim client to eval policy:
```bash
source examples/libero/.venv/bin/activate
export LIBERO_ROOT=$(pwd)/../LIBERO
export PYTHONPATH=$PYTHONPATH:$LIBERO_ROOT
python examples/libero/eval_libero_policy.py host=127.0.0.1 port=8000 libero_task_suite=libero_spatial headless=false action_chunk_size=5 episodes_per_task=10
```
