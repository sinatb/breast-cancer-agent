"""The smoke run: a scripted stand-in for the LLM drives the real loop and tools, with training stubbed here."""

from langchain_core.messages import HumanMessage

from agent.Orchestrator import Orchestrator
from agent.smoke import SMOKE_EXPERIMENTS, SMOKE_SAMPLES_PER_CLASS, ScriptedLLM, smoke_llm, smoke_passed
from dataset.splits import sample_per_class, split_dataframe
from main import parse_args


def test_smoke_run_trains_two_models_for_two_epochs_and_finishes(training_stubs, settings):
    orchestrator = Orchestrator(settings, llm=smoke_llm(), samples_per_class=SMOKE_SAMPLES_PER_CLASS)

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert result.experiments == 2
    assert len(set(training_stubs.built_models)) == 2
    assert [call["epochs"] for call in training_stubs.fine_tune_calls] == [2, 2]
    assert training_stubs.loader_kwargs == [{"samples_per_class": SMOKE_SAMPLES_PER_CLASS}] * 2
    assert smoke_passed(result)


def test_smoke_run_fails_when_an_experiment_fails(training_stubs, settings):
    training_stubs.fine_tune_error = RuntimeError("no weights")
    orchestrator = Orchestrator(settings, llm=smoke_llm(), samples_per_class=SMOKE_SAMPLES_PER_CLASS)

    result = orchestrator.run()

    assert result.experiments == 0
    assert not smoke_passed(result)


def test_smoke_experiments_are_valid_tool_input(training_stubs, settings):
    orchestrator = Orchestrator(settings, llm=smoke_llm())

    for experiment in SMOKE_EXPERIMENTS:
        assert not orchestrator.tools_by_name["run_experiment"].invoke(experiment).startswith("ERROR")


def test_scripted_llm_finishes_once_the_script_is_over():
    llm = ScriptedLLM([{"name": "check_budget_status", "args": {}}])
    history = [HumanMessage(content="go")]

    first, second = llm.invoke(history), llm.invoke(history)

    assert [call["name"] for call in first.tool_calls] == ["check_budget_status"]
    assert [call["name"] for call in second.tool_calls] == ["finish"]
    assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"]


def test_smoke_sample_leaves_every_class_in_training(labels_df):
    train_df, val_df, _ = split_dataframe(sample_per_class(labels_df, SMOKE_SAMPLES_PER_CLASS))

    assert set(train_df["label"]) == {1, 2, 3, 4, 5}
    assert len(val_df) > 0


def test_smoke_flag():
    assert parse_args([]).smoke is False
    assert parse_args(["--smoke"]).smoke is True
