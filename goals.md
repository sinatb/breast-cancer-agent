# Project Goals

The code review and improvement plan for the breast cancer fine-tuning agent. Goals are ordered: each one depends on the
ones before it, and in particular G3 (fair evaluation) comes before G4 (better models), so improvements can be measured.

## 1. Assessment

| Dimension | Score | Why |
|---|---|---|
| Architecture | **6/10** | Good bones for about 1,000 lines. The packages have clear jobs, the adapter and catalog make models pluggable, and `create_agent_tools(catalog, budget, memory)` injects its dependencies cleanly. It loses points for configuration read at import time, no run or config object, no evaluation layer, no persistence, and an "agent" that is not a loop. |
| Cleanness | **5/10** | Small readable functions and reasonable type hints. It loses points for no tests, `print` as logging, dead code and imports, the same warning filter copied into four files, the magic number 5 for classes in several places, three different batch-size defaults, and module names that break Python convention. |
| **Overall** | **5.5/10** | A solid course-project structure that needs hardening. |

Outside that score, **experimental validity is about 3/10**, and this is the most important problem in the project.
There is no seeding, the test set is never used, and splits are not stratified. Checkpoints are chosen by accuracy on
100 images, 64 of which are class 1. LoRA also trains with a frozen random head. The two committed results (accuracy
0.58 and 0.48) are both below the 0.64 that always predicting class 1 would score on that validation set, so the
current results cannot support any conclusion about which model or strategy is better.

### How this was checked

This review is static. **The project was not executed**, because the dataset and API keys are not available in this
checkout and PyTorch was deliberately not installed for the review. Findings are marked:

- **Confirmed**: proven by the code itself, by committed notebook output, by arithmetic, or by the PyTorch source.
- **To verify**: strongly suspected, needs one run to prove. G1 includes these checks.

## 2. Findings

### Critical: results are wrong or the project does not run

| # | Problem | Where | Status |
|---|---|---|---|
| C1 | **LoRA freezes the classification head.** PEFT marks only `lora_` parameters trainable and nothing re-enables the head. The notebook prints 1,769,472 trainable parameters for ViT, which is exactly the LoRA parameters alone (LoRA plus head would be 1,773,317). Every LoRA run so far classified through a random linear layer. | [FineTuner.py:130](finetuning/FineTuner.py#L130) | Confirmed |
| C2 | **Some LoRA targets do nothing.** ViT `self_attention.in_proj_weight` is a parameter, not a module, so it is never matched. `self_attention.out_proj` is matched, but `MultiheadAttention` reads `out_proj.weight` directly and never calls the wrapped module, so those LoRA weights never affect the output. | [ModelAdapter.py:27](finetuning/ModelAdapter.py#L27) | Confirmed |
| C3 | Swin targets `mlp.fc1` and `mlp.fc2`, but torchvision's Swin MLP is a `Sequential` named `mlp.0` and `mlp.3`. MobileNet's `block.N.0` patterns may land on depthwise convolutions or miss entirely. Nothing fails when a pattern matches nothing. | [ModelAdapter.py:39](finetuning/ModelAdapter.py#L39), [ModelAdapter.py:75](finetuning/ModelAdapter.py#L75) | To verify |
| C4 | **Linear probe is not really frozen.** `_train_epoch` calls `model.train()` on the whole model, so BatchNorm statistics in the "frozen" backbone keep changing for ResNet, DenseNet and MobileNet. | [FineTuner.py:168](finetuning/FineTuner.py#L168) | Confirmed |
| C5 | **Environment is incomplete.** `environment.yaml` does not declare `peft`, `langchain-openai`, `python-dotenv`, `scikit-learn`, `scikit-image` or `scipy`, all of which are imported. | [environment.yaml](environment.yaml) | Fixed (replaced by `pyproject.toml` + `uv.lock`) |
| C6 | **Import crashes without configuration.** `os.path.join(None, "images")` raises `TypeError` at import when `DATASET_PATH` is unset, and there is no `.env.example` or list of required variables. If `MODEL_DIR` is unset, checkpoints silently go to a folder literally named `None/`. | [preprocess.py:13](dataset/preprocess.py#L13), [FineTuner.py:320](finetuning/FineTuner.py#L320) | Confirmed |
| C7 | **One tool failure ends the session and loses everything.** Parallel tool calls raise `RuntimeError`, and a CUDA out-of-memory error or bad argument propagates uncaught. `ExperimentMemory` lives only in RAM and `history.txt` is written only at the very end. | [Orchestrator.py:171](agent/Orchestrator.py#L171), [main.py:24](main.py#L24) | Confirmed |
| C8 | **The agent is not a loop.** `invoke` makes one LLM call, runs one tool, and returns. The model never sees the result in that turn, so the "Analyze, then act" flow depends on a human pressing keys, and a step that only checks the budget still costs a whole turn. `input()` also prevents unattended runs. | [Orchestrator.py:164](agent/Orchestrator.py#L164), [main.py:10](main.py#L10) | Confirmed |
| C9 | **Notebooks do not import as committed.** `from FineTuner import ...` and `from DataLoader import *` load modules that use relative imports, which raises `ImportError` outside a package. | [finetuning_analysis.ipynb](finetuning/finetuning_analysis.ipynb), [dataset analysis.ipynb](dataset/dataset%20analysis.ipynb) | Confirmed |

### Evaluation validity

| # | Problem | Where | Status |
|---|---|---|---|
| E1 | No seeding of Python, NumPy, PyTorch or DataLoader workers, so run-to-run noise is unknown and differences between runs cannot be interpreted. | whole project | Confirmed |
| E2 | Splits are not stratified. Validation has 64/21/9/4/2 samples per class where proportional would be about 60/21/7/7/5, so class 5 is judged on 2 images. | [DataLoader.py:24](dataset/DataLoader.py#L24) | Confirmed |
| E3 | `test_loader` is created and never used. No reported number comes from held-out data. | [tools.py:66](agent/tools.py#L66) | Confirmed |
| E4 | Checkpoints are selected by accuracy while the loss is class-weighted. These goals conflict: accuracy rewards ignoring the rare classes that the weights try to protect. | [FineTuner.py:101](finetuning/FineTuner.py#L101) | Confirmed |
| E5 | The budget counts experiments, so 20 epochs of full ViT costs the same as 1 epoch of a MobileNet probe. The prompt's advice to "conserve budget" with small models has no effect under this budget. | [AgentBudget.py](agent/AgentBudget.py) | Confirmed |
| E6 | All strategies share the default learning rate and fixed epochs in the notebook, which favors whichever strategy that setting happens to suit. | [finetuning_analysis.ipynb](finetuning/finetuning_analysis.ipynb) | Confirmed |
| E7 | Images may be from the same patient (for example two views), which would leak across splits. `train_labels.csv` has only a filename and a label. | dataset | To verify |
| E8 | There is no baseline, either for the classifier (majority class) or for the agent (random search). Whether the LLM adds value is untested. | whole project | Confirmed |

### Other bugs and quality issues

| # | Problem | Where |
|---|---|---|
| Q1 | The system prompt names a tool `get_experience`, but the real name is `get_experience_summary`. | [Orchestrator.py:30](agent/Orchestrator.py#L30) |
| Q2 | The prompt tells the agent to compare train and validation accuracy, but training accuracy is never computed or returned. The confusion matrix is also missing from `get_experience_summary`. | [Orchestrator.py:81](agent/Orchestrator.py#L81), [FineTuneResult.py:19](finetuning/FineTuneResult.py#L19) |
| Q3 | Batch size defaults to 8 in the schema, 32 in the tool function, and 64 in the loader. | [tools.py:26](agent/tools.py#L26), [tools.py:46](agent/tools.py#L46), [DataLoader.py:20](dataset/DataLoader.py#L20) |
| Q4 | Tool inputs are barely validated: epochs of 0 or fewer, negative learning rates and very large batch sizes are all accepted. Zero epochs returns a result of `-inf`. | [tools.py:61](agent/tools.py#L61) |
| Q5 | The model and data are built before the budget is checked, and repeating a config overwrites its checkpoint folder despite the "never repeat" prompt rule. | [tools.py:64](agent/tools.py#L64) |
| Q6 | `_compute_class_weights` runs the full image pipeline over the training set just to count labels, once per experiment. | [FineTuner.py:143](finetuning/FineTuner.py#L143) |
| Q7 | Watershed preprocessing runs on every image, every epoch, with no cache. This is likely the main CPU bottleneck. | [BreastCancerDataset.py:18](dataset/BreastCancerDataset.py#L18) |
| Q8 | Training has no augmentation (train and test transforms are identical), which fits the overfitting in the ResNet run: train loss 0.30 against validation loss 2.57 by epoch 5. | [DataLoader.py:37](dataset/DataLoader.py#L37) |
| Q9 | `images.requires_grad_(True)` computes gradients for input pixels for no benefit. | [FineTuner.py:177](finetuning/FineTuner.py#L177) |
| Q10 | Each checkpoint stores the full model and optimizer, about 1 GB for ViT under `FULL_FT`. `model_dir/` and `history.txt` are also missing from `.gitignore`. | [FineTuner.py:267](finetuning/FineTuner.py#L267) |
| Q11 | `index > len(self.df)` is off by one. The filename parsing `[6:10]` assumes exactly four digits. | [BreastCancerDataset.py:15](dataset/BreastCancerDataset.py#L15) |
| Q12 | Crops are resized straight to a square, which distorts anatomy by a different amount in each image. White-background images may defeat the fixed threshold of 5 (to verify). | [preprocess.py:23](dataset/preprocess.py#L23), [preprocess.py:61](dataset/preprocess.py#L61) |
| Q13 | Dead or misplaced code: `TaskType`, `List`, `load_checkpoint`, `frozen_backbone`, `torchaudio`, the one-line `utils/` wrapper, plotting inside `DataLoader.py`, and `verification.py` posing as a test. | various |
| Q14 | The LLM model, budget of 10, epoch cap of 20, LoRA settings and class count are hard-coded, and the number 10 also appears in the prompt and a print statement. | [Orchestrator.py:138](agent/Orchestrator.py#L138) |

## 3. Goals

### G1. Make it run correctly

*Fixes C1–C9 and Q1–Q6, Q9–Q11. Target: a clean clone runs end to end from documented steps.*

**Environment and configuration**
- [x] Add a `pyproject.toml` that declares every direct dependency with version ranges, managed with uv: a CPU torch group for local runs, a CUDA group for GPU machines, and Kaggle reusing its preinstalled torch.
- [ ] Add `.env.example` documenting `DATASET_PATH`, `MODEL_DIR`, `API_KEY`, `BASE_URL` and the LLM model name.
- [ ] Add a typed `config.py`, and remove every `load_dotenv()` and `os.getenv()` that runs at import time. Paths are resolved when a run starts, with a clear error naming any missing setting.
- [ ] Replace the copied warning filters by fixing the cause (`setuptools` pin or a newer torch).

**Training correctness**
- [ ] Keep the head trainable under LoRA.
- [ ] Add one adapter-level assertion that every LoRA target pattern matches at least one module that is actually called during `forward`.
- [ ] Correct the ViT, Swin and MobileNet targets. For ViT attention, either accept MLP-only LoRA and document it, or move ViT and Swin to `timm`, whose attention layers are plain `Linear` modules.
- [ ] Keep frozen layers in eval mode during training, so BatchNorm statistics do not change for frozen backbones.
- [ ] Count class weights from the dataframe, remove `requires_grad_` on inputs, and centralize the conversion from 1-based labels to 0-based.
- [ ] Save only trainable parameters and metadata, in a unique folder per run.

**Agent loop**
- [ ] Implement a real loop: call the LLM, run the tool, feed back the result, and repeat until a `finish` tool, an exhausted budget, or a maximum step count.
- [ ] Pass `parallel_tool_calls=False` to `bind_tools`. As a fallback for providers that ignore it, run the first call and answer the others with an error `ToolMessage` instead of raising.
- [ ] Catch tool exceptions and return them to the model as errors. Clear GPU memory between experiments.
- [ ] Validate tool inputs in the schema using `Literal` types and numeric bounds, and fix the tool name in the prompt.
- [ ] Add an unattended mode, keeping human approval as an option.

**Done when:** a new user can follow the README and complete a smoke run (2 models, 2 epochs, a stub LLM, a small image sample) with no code edits. The C3, E7 and Q12 checks are recorded in this file. Notebooks run top to bottom after restarting the kernel.

### G2. Run on Kaggle and Colab

*Target: the same commit runs on local, Kaggle and Colab with no code changes, and survives a killed session.*

- [ ] Add `runtime.py` to detect the platform (`KAGGLE_KERNEL_RUN_TYPE`, `COLAB_RELEASE_TAG`) and supply defaults:

| | Local | Kaggle | Colab |
|---|---|---|---|
| Data (read) | `.env` path | `/kaggle/input/<dataset>` (read-only) | Unzip from Drive to `/content` (avoids slow Drive reads) |
| Fast cache | local folder | `/kaggle/working/cache` | `/content/cache` |
| Durable output | local folder | `/kaggle/working/runs` (limited, about 20 GB) | `/content/drive/MyDrive/...` |
| Secrets | `.env` | `kaggle_secrets.UserSecretsClient` | `google.colab.userdata` |

- [ ] Settings resolve in this order: explicit argument, environment variable, platform default.
- [ ] Add one `notebooks/run_cloud.ipynb` that works on both platforms: clone or attach the code, install, check the GPU, then run a chosen profile. It must work under Kaggle's "Save & Run All", so no `input()`.
- [ ] Don't pin `torch`, `torchvision` or `numpy` on hosted runtimes. Install only the missing packages, to avoid breaking their preinstalled builds.
- [ ] Add a one-time preprocessing step that caches images to disk (fixes Q7). Size the DataLoader from `os.cpu_count()` instead of the hard-coded 4 workers, and enable mixed precision.
- [ ] **Resume:** write each completed run and each agent step to an append-only JSONL file as it happens. Restarting skips finished work and restores the budget and agent history.
- [ ] Handle no-internet Kaggle sessions: load pretrained weights from an attached Kaggle dataset (`TORCH_HOME`), and give a clear error if the LLM endpoint is unreachable.
- [ ] Use exactly one GPU even when Kaggle provides two, so results stay comparable.
- [ ] Publish run profiles: `smoke`, `lite` (fits one free session), and `full`.
- [ ] Document how to attach the dataset, after checking its license allows uploading it.

**Done when:** the smoke profile passes on both Kaggle and Colab from one notebook, and a session deliberately killed mid-run resumes without losing or repeating a finished experiment.

### G3. Fair and meaningful evaluation

*Target: one evaluation protocol applied identically, so that a stated difference between modes is a real difference.*

**What is compared.** "Modes" means three levels, and each needs its own question:

| Level | Question | Compared |
|---|---|---|
| A. Fine-tuning strategy | Which strategy works best for a given model? | `FULL_FT`, `LINEAR_PROBE`, `LORA` |
| B. Architecture | Which model works best, and at what cost? | The five catalog models |
| C. Search method | Does the LLM agent find better models than simple search? | Agent, random search, grid search, scripted policy, successive halving |

**Protocol (fixed before any comparison is run)**
- [ ] **Splits:** a locked, stratified 20% test set (200 images), plus stratified 5-fold cross-validation on the remaining 800. Save it once as a versioned `splits.json`, and record its hash with every run. Group by patient if IDs can be found, and check for duplicate images.
- [ ] **Test lock:** the agent and all tuning code can only read validation data. Test evaluation happens through one command that logs each use, once per final pipeline.
- [ ] **One selection metric everywhere:** macro-F1 on validation decides checkpoints, hyperparameters and the agent's objective. If the labels are confirmed ordinal, quadratic weighted kappa should replace it. Choose this before looking at results.
- [ ] **Equal tuning effort:** every (model, strategy) pair receives the same number of trials over a learning-rate range suited to that strategy, with the same maximum epochs and early-stopping rule. Tune on fold 0, then compare on folds 1–4.
- [ ] **Paired runs:** every mode uses the same folds, seeds (at least 3), preprocessing, augmentation and class weighting.
- [ ] **Baselines:** report a majority-class baseline (accuracy about 0.60–0.64, macro-F1 about 0.16) and a stratified-random baseline in every table. A mode that does not beat them is reported as such.

**Reporting**
- [ ] Metrics: macro-F1, balanced accuracy, per-class recall and precision, confusion matrix, one-vs-rest AUROC, quadratic weighted kappa, calibration error, and sensitivity for the high-risk group once the classes are defined. Accuracy is secondary.
- [ ] Uncertainty: report mean and standard deviation across folds and seeds, plus 95% bootstrap intervals on pooled out-of-fold predictions. Class 5 has only 48 images, so expect wide intervals and say so.
- [ ] Significance: use a paired bootstrap on the same out-of-fold predictions, with Holm correction for multiple comparisons. Don't rely on a Wilcoxon test over 5 folds, since its smallest possible two-sided p-value is 0.0625.
- [ ] Cost beside quality: wall time, epochs used, trainable parameters, GFLOPs, peak GPU memory.
- [ ] Every run is one row in a results file with its commit, config hash, split hash, seed and platform. Generate the tables and plots from that file with one command, with no hand-copied numbers.

**Agent against search (level C)**
- [ ] Replace the experiment-count budget with a cost budget, for example `epochs × GFLOPs`. It is deterministic, so it does not vary between Kaggle and Colab.
- [ ] Give every method the same search space, the same budget and the same feedback.
- [ ] Precompute a results table covering the discretized search space, then let each method query it (the NAS-Bench approach). This allows at least 20 repeats per method at almost no GPU cost, which is necessary because LLM runs vary even at temperature 0.
- [ ] Separate prior knowledge from reasoning. The current prompt effectively scripts the search, so evaluate a guided prompt and a neutral prompt, and include a scripted baseline that follows the guided prompt's rules without an LLM.
- [ ] Report best-so-far against budget spent, final regret against the best config in the table, the locked-test score of each method's final choice, and LLM tokens, cost and invalid tool calls.

**Done when:** one command rebuilds the comparison report from the results file. Each claim of the form "X beats Y" has an interval and a corrected p-value. The report answers levels A, B and C, including if the answer is "no measurable difference" or "the agent does not beat random search".

### G4. Improve model performance

*Addresses "fixing the low accuracy and precision problem" from the README. Each change is accepted only if it improves G3's metric beyond its interval.*

- [ ] Review a gallery of preprocessing outputs and fix failures, especially white backgrounds and remaining laterality markers.
- [ ] Preserve aspect ratio by padding, and test higher resolutions (for example 384 or 512) where the model allows.
- [ ] Add mild augmentation for mammography: small rotations, scale, brightness and contrast. Avoid horizontal flips unless justified, because preprocessing already normalizes orientation.
- [ ] Compare imbalance methods: softened class weights, focal loss, balanced sampling. The current inverse-frequency weights differ by a factor of about 12.
- [ ] Try ordinal losses if the labels are ordinal, and a binary high-risk task if that is the clinically useful question.
- [ ] Add ensembling (from the README's further work) and test-time augmentation.
- [ ] Use Grad-CAM to confirm that the model looks at breast tissue rather than markers or borders.

**Done when:** the final pipeline beats the G1 baseline on locked-test macro-F1 with non-overlapping or significantly different intervals, or the report concludes that 1,000 images cannot support further gains.

### G5. Make the agent a useful decision-maker

*Target: decisions based on evidence the agent can actually see.*

- [ ] Return per-class metrics, the train-versus-validation gap, best epoch, and remaining cost after every experiment, in structured form.
- [ ] Add tools for ranked results and for comparing two runs, and enforce the no-duplicates rule in code.
- [ ] Let the agent stop early with a stated reason, and finish with a recommendation that names its evidence.
- [ ] Move prompts into their own module, generated from configuration so that budgets and tool names cannot drift out of date.
- [ ] Make the LLM provider and model configurable, and record a full trace of each decision as JSONL.

**Done when:** a stub-LLM test covers the full loop including tool errors, parallel calls, budget exhaustion and resume, and G3's level C shows whether these changes improved search results.

### G6. Engineering quality

*Target: the architecture and cleanness scores both reach at least 8/10.*

Target layout, with `snake_case` modules per PEP 8:

```text
breast_cancer_agent/
  config.py  runtime.py
  data/        splits.py  preprocess.py  dataset.py  cache.py
  models/      catalog.py  adapters.py  strategies.py
  training/    trainer.py  metrics.py  checkpoints.py
  agent/       orchestrator.py  tools.py  budget.py  memory.py  prompts.py
  evaluation/  baselines.py  search.py  stats.py  report.py
notebooks/
tests/
```

- [ ] Add a CLI: `python -m breast_cancer_agent {preprocess,train,agent,evaluate,report}`.
- [ ] Add `pytest` coverage for: stratified, non-overlapping splits; LoRA trainable counts per model including the head; frozen BatchNorm stability; metric correctness; budget accounting; resume. Use small random tensors, so no dataset is needed.
- [ ] Add GitHub Actions running `ruff` and CPU tests.
- [ ] Replace `print` with `logging`, remove the dead code from Q13, and define `num_classes` once.
- [ ] Update `.gitignore` for run outputs, and move notebooks to `notebooks/` with names that contain no spaces.
- [ ] Rewrite the README around setup, running and results, with figures generated by G3.

**Done when:** CI passes on every push, no module reads the environment at import, and there is no hard-coded experiment setting outside the configuration.

### G7. Later, once G1–G6 are done

- [ ] A small UI to watch runs and the agent's reasoning (from the README's further work).
- [ ] A model card: data, intended use, per-class performance, and known limits. State clearly that this is a research project and not a diagnostic tool.
- [ ] Validation on an external mammography dataset to test generalization.

## 4. Order of work

| Milestone | Goals | Result |
|---|---|---|
| M1 | G1 | Correct, runnable project on the smoke profile |
| M2 | G2, the G3 split and seeding work, and the core of G6 (config, tests) | Reproducible and resumable on cloud GPUs |
| M3 | G3 | First trustworthy baseline for strategies and models. **Earlier results should be discarded here.** |
| M4 | G4, G5 | Better models and a better agent, each measured against M3 |
| M5 | G6 remainder, G7 | Final report and public-ready repository |

## 5. Open questions

These change the plan and can only be answered by the project owner:

1. **What do labels 1–5 mean?** If they are BI-RADS-like and ordered, this affects the metric, the loss and the high-risk group definition.
2. **Where does the dataset come from, and are patient IDs available?** This decides whether group splitting is possible (E7) and whether the data may be uploaded to Kaggle or Colab.
3. **How much GPU time is available?** This decides the `full` profile. The full G3 design is about 180 comparison runs (15 model-strategy pairs × 4 folds × 3 seeds) plus tuning, so measure time per run after G2's caching before committing to it.
4. **Will the LLM provider remain Metis?** Confirm that it honors `parallel_tool_calls=False` and is reachable from Kaggle and Colab.
