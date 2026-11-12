"""Agent tools: input validation, budget accounting and cleanup around an experiment.

Model building, data loading and training are stubbed (see ``training_stubs``); nothing is trained or downloaded.
"""

import pytest
from pydantic import ValidationError

from agent.AgentBudget import AgentBudget
from agent.ExperimentMemory import ExperimentMemory
from agent.tools import DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE, MAX_EPOCHS, create_agent_tools
from finetuning.FineTuneStrategy import FineTuneStrategy
from finetuning.ModelCatalog import get_default_catalog

VALID_ARGS = {
    "model_name": "resnet50",
    "strategy": "LORA",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 2,
    "rationale": "Baseline LoRA run on a mid-sized CNN.",
}


@pytest.fixture
def budget() -> AgentBudget:
    return AgentBudget(max_experiments=2)


@pytest.fixture
def memory() -> ExperimentMemory:
    return ExperimentMemory()


@pytest.fixture
def tools(training_stubs, settings, budget, memory) -> dict:
    created = create_agent_tools(catalog=get_default_catalog(), budget=budget, memory=memory, settings=settings)
    return {tool.name: tool for tool in created}


def test_tool_names(tools):
    assert set(tools) == {
        "get_available_models",
        "check_budget_status",
        "run_experiment",
        "get_experience_summary",
        "finish",
    }


def test_run_experiment_seeds_before_building_the_model(tools, training_stubs, monkeypatch):
    """E1: the head is initialised when the model is built, so seeding must come first."""
    import agent.tools as agent_tools

    seeded_at = []
    monkeypatch.setattr(agent_tools, "seed_everything", lambda seed: seeded_at.append(
        (seed, len(training_stubs.built_models))), raising=False)

    tools["run_experiment"].invoke(VALID_ARGS)

    assert seeded_at == [(agent_tools.SEED, 0)]


def test_run_experiment_trains_and_records(tools, training_stubs, budget, memory):
    summary = tools["run_experiment"].invoke(VALID_ARGS)

    assert "ERROR" not in summary
    assert training_stubs.built_models == ["resnet50"]
    assert training_stubs.fine_tune_calls[0]["strategy"] is FineTuneStrategy.LORA
    assert training_stubs.fine_tune_calls[0]["epochs"] == 2
    assert budget.conducted_experiments == 1
    assert len(memory.conducted_experiments) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"model_name": "resnet18"},
        {"strategy": "lora"},
        {"strategy": "PREFIX_TUNING"},
        {"epochs": 0},
        {"epochs": -3},
        {"epochs": MAX_EPOCHS + 1},
        {"learning_rate": 0.0},
        {"learning_rate": -1e-4},
        {"learning_rate": 5.0},
        {"weight_decay": -0.1},
        {"batch_size": 0},
        {"batch_size": MAX_BATCH_SIZE + 1},
        {"rationale": ""},
    ],
    ids=lambda override: "{}={}".format(*next(iter(override.items()))),
)
def test_run_experiment_rejects_invalid_input(tools, training_stubs, budget, override):
    with pytest.raises(ValidationError):
        tools["run_experiment"].invoke({**VALID_ARGS, **override})

    assert training_stubs.built_models == []
    assert training_stubs.fine_tune_calls == []
    assert budget.conducted_experiments == 0


def test_schema_lists_the_allowed_models_and_strategies(tools):
    properties = tools["run_experiment"].tool_call_schema.model_json_schema()["properties"]

    assert properties["model_name"]["enum"] == list(get_default_catalog().model_dict)
    assert set(properties["strategy"]["enum"]) == {strategy.name for strategy in FineTuneStrategy}
    assert properties["epochs"]["maximum"] == MAX_EPOCHS


def test_batch_size_has_one_default(tools, training_stubs):
    properties = tools["run_experiment"].tool_call_schema.model_json_schema()["properties"]

    tools["run_experiment"].invoke(VALID_ARGS)

    assert properties["batch_size"]["default"] == DEFAULT_BATCH_SIZE
    assert training_stubs.loader_batch_sizes == [DEFAULT_BATCH_SIZE]


def test_exhausted_budget_is_refused_before_anything_is_built(tools, training_stubs, budget):
    budget.conducted_experiments = budget.max_experiments

    result = tools["run_experiment"].invoke(VALID_ARGS)

    assert result.startswith("ERROR")
    assert training_stubs.built_models == []
    assert training_stubs.loader_batch_sizes == []
    assert training_stubs.fine_tune_calls == []


def test_memory_is_released_after_a_successful_experiment(tools, training_stubs):
    tools["run_experiment"].invoke(VALID_ARGS)

    assert training_stubs.memory_releases == 1


def test_failed_experiment_releases_memory_and_costs_nothing(tools, training_stubs, budget, memory):
    training_stubs.fine_tune_error = RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        tools["run_experiment"].invoke(VALID_ARGS)

    assert training_stubs.memory_releases == 1
    assert budget.conducted_experiments == 0
    assert memory.conducted_experiments == []


def test_agent_sees_training_accuracy_and_the_confusion_matrix(tools):
    """Q2: both were missing from what the tools return."""
    result = tools["run_experiment"].invoke(VALID_ARGS)
    experience = tools["get_experience_summary"].invoke({})

    assert "Train Acc: 0.8750" in result
    assert "[[7, 1], [2, 5]]" in result
    assert "0.875" in experience
    assert "[[7, 1], [2, 5]]" in experience


def test_all_images_are_used_by_default(tools, training_stubs):
    tools["run_experiment"].invoke(VALID_ARGS)

    assert training_stubs.loader_kwargs == [{"samples_per_class": None}]


def test_sample_size_reaches_the_loaders(training_stubs, settings, budget, memory):
    created = create_agent_tools(
        catalog=get_default_catalog(), budget=budget, memory=memory, settings=settings, samples_per_class=3
    )
    run_experiment = {tool.name: tool for tool in created}["run_experiment"]

    run_experiment.invoke(VALID_ARGS)

    assert training_stubs.loader_kwargs == [{"samples_per_class": 3}]


def test_finish_acknowledges_the_reason(tools):
    assert "budget spent" in tools["finish"].invoke({"reason": "budget spent"})


def test_run_experiment_requires_a_rationale(tools, training_stubs):
    args = {key: value for key, value in VALID_ARGS.items() if key != "rationale"}

    with pytest.raises(ValidationError):
        tools["run_experiment"].invoke(args)

    assert training_stubs.fine_tune_calls == []


def test_agent_sees_macro_f1_and_per_class_recall(tools):
    """The stub's confusion matrix [[7, 1], [2, 5]] gives recall 7/8 and 5/7 per true class (rows)."""
    result = tools["run_experiment"].invoke(VALID_ARGS)
    experience = tools["get_experience_summary"].invoke({})

    assert "Val macro-F1: 0.2000" in result
    assert "Per-class recall: [0.88, 0.71]" in result
    assert "Val F1:" not in result
    assert "macro_f1 : 0.2000" in experience
