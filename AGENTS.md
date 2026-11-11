# AGENTS.md

An LLM agent that fine-tunes vision models (ViT, Swin, ResNet, DenseNet, MobileNetV3) for 5-class breast cancer
classification. It uses linear-probe, full fine-tuning and LoRA strategies. `goals.md` holds the roadmap, the known
bugs (C1–C6) and the order of work. Read it before any non-trivial change.

## Commands

```bash
uv sync                      # local env: CPU torch + dev tools
uv run pytest                # unit tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run python main.py        # run the agent (needs .env, see below)
```

- Kaggle setup is in `README.md`. Never add a Kaggle-only dependency to the base list.
- Add dependencies with `uv add <pkg>` or `uv add --group dev <pkg>`. Never use `pip install`.
- torch and torchvision belong only in the `cpu` and `gpu` groups.

## Configuration

Copy `.env.example` to `.env`, which is never committed. `config.load_settings()` reads it when a run starts.
Environment variables override `.env`, and relative paths resolve against the project root.

| Variable | Purpose |
|---|---|
| `DATASET_PATH` | Folder with `images/` and `train_labels.csv` (required) |
| `MODEL_DIR` | Where checkpoints are written (required) |
| `API_KEY` | Key for the LLM endpoint (required by the agent only) |
| `BASE_URL` | OpenAI-compatible LLM endpoint (optional, OpenAI default) |
| `LLM_MODEL` | LLM model name (optional, default `gpt-4.1-mini`) |

## Layout

- `dataset/`: splitting, preprocessing and the `Dataset` and `DataLoader` setup.
- `finetuning/`: the model catalog, per-architecture adapters (head and LoRA targets), strategies and the training loop.
- `agent/`: the LangChain orchestrator, its tools, the budget and the experiment memory.
- `utils/`: small model helpers.
- `tests/`: pytest unit tests.

## Rules

- **The test split is for final evaluation only.** Never use it for model selection, tuning or anything the agent can see.
- **Never weaken, skip or delete a test to make it pass.** If a test looks wrong, stop and explain why.
- **Unit tests must not need the real dataset, a GPU, the network or the LLM.** Use tiny DataFrames, random tensors and stubbed LLM responses.
- **When fixing a bug, first add a test that fails because of it.**
- **Don't read the environment at import time.** Resolve configuration when a run starts.
- **New modules use `snake_case.py`.** Don't rename existing modules unless the task asks for it.
- **No secrets, data, checkpoints or run outputs in git.**
- **Keep changes scoped to the task.** Mention unrelated problems you notice instead of fixing them.

## Definition of done

- `uv run pytest` and `uv run ruff check .` pass.
- New behavior has a test.
- The summary says what changed, why, and how it was verified, including anything that could not be run.
