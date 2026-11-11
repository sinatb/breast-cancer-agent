# Breast Cancer Agent
This is the report for the final project of the Deep Learning course at IUST. In this project a Breast Cancer detection
agent has been implemented which fine-tunes different models form a model catalog with 3 different strategies. This 
report has been written in a hierarchical fashion. Meaning this file is just a starting point, In depth analysis of each
part is done in the respective notebook.

## Setup
Dependencies are managed with [uv](https://docs.astral.sh/uv/) through `pyproject.toml` and `uv.lock`.

**Local (CPU).** `uv sync` creates `.venv` with the CPU build of PyTorch, plus `ipykernel` for the notebooks.
```bash
uv sync
uv run python verification.py   # expect "CUDA available: False"
cp .env.example .env            # then put the dataset in data/ (or point DATASET_PATH at it)
uv run python main.py --smoke   # check the setup, no API key needed
```

The dataset folder needs `images/` and `train_labels.csv`. Settings are described in `.env.example`; the agent also
needs `API_KEY`, while the smoke run and the notebooks do not.

**Kaggle (GPU).** Kaggle already ships a CUDA build of PyTorch, so only the other dependencies are installed, into
the system Python:
```bash
!pip install -q uv
!uv pip install --system -r pyproject.toml
!python verification.py          # expect "CUDA available: True"
```

**Other GPU machines (CUDA 12.6).** Install with the `gpu` group instead of the default `cpu` group:
```bash
uv sync --no-default-groups --group gpu --group dev
```

## Running the agent
`uv run python main.py` runs the agent unattended: it calls the LLM, runs the tool the LLM chose, feeds the result
back and repeats. The session ends when the agent calls its `finish` tool, when the experiment budget is spent, or when
the step limit is reached. A failed tool call is reported to the agent instead of ending the session.

| Flag | Meaning |
|---|---|
| `--approve` | Ask before each experiment. Answering anything but `y` ends the session. |
| `--max-experiments N` | Experiment budget (default 10). |
| `--max-steps N` | Maximum number of LLM replies (default 40). |
| `--smoke` | Check the setup instead: a scripted stand-in for the LLM trains MobileNetV3 and ResNet-50 for 2 epochs each on 16 images per class. It needs no API key and takes a few minutes on a CPU. It ends with `Smoke run passed.`, or exits with an error. Its numbers mean nothing. |

The chat history is written to `history.txt` inside `MODEL_DIR` when the session ends, including after a crash.
The first run downloads pretrained weights (about 125 MB for the smoke run).

The notebooks read the same `.env`. To check that `finetuning_analysis.ipynb` runs without training for hours, set
`QUICK_CHECK = True` in its first cell.

## Overall Architecture
This project consists of 3 packages. `dataset`, `agent` and `finetuning`. The `utils` package contains some simple
utilities for getting a model.

The `dataset` package controls how the data is split between train, validation and test
sets. The preprocessing part of this project is also done in the `dataset` package. For more information about the 
nooks and crannies of the implementation and overall challenges I suggest reading `dataset_analysis.ipynb`.

The `finetuning` package implements simple finetuning utility used by the agent down the line. This includes, simple
train and validation loops and the creation of `BreastCancerModel`. This model is actually something I am quite proud of,
so I suggest reading `finetuning_analysis.ipynb` to gain a better understanding of how the package works.

The final package is `agent`. In this package the tools and the orchestrator have been implemented. To gain better insight
of this package please refer to `agent_analysis.md`.

## Further Work (done before making repo public)
1. Adding ensemble support.
2. Fixing the low accuracy and precision problem.
3. Adding UI for the chat and the overall system.
4. Creating the final model for test.