"""The agent loop, driven by a scripted stub LLM: feedback, stop conditions and failure handling.

No LLM endpoint, dataset or training is involved; the tools are real but everything behind them is stubbed.
"""

import re

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.Orchestrator import Orchestrator
from config import ConfigError

EXPERIMENT_ARGS = {
    "model_name": "resnet50",
    "strategy": "LINEAR_PROBE",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 2,
    "rationale": "Cheap linear probe to get a first baseline.",
}


def call(name: str, args: dict | None = None, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args or {}, "id": call_id}


def reply(*tool_calls: dict, content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(tool_calls))


class StubLLM:
    """Replays scripted replies and records how it was bound and what it was shown."""

    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.seen: list[list] = []
        self.bind_kwargs: dict | None = None
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools, **kwargs):
        self.bind_kwargs = kwargs
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        if self.replies:
            next_reply = self.replies.pop(0)
            if isinstance(next_reply, Exception):
                raise next_reply
            return next_reply
        return AIMessage(content="Still thinking.")


def rejected_tool_call() -> openai.BadRequestError:
    """The 400 the provider returned when gpt-oss leaked `<|channel|>commentary` into a tool name."""
    message = "attempted to call tool 'run_experiment<|channel|>commentary' which was not in request.tools"
    response = httpx.Response(400, request=httpx.Request("POST", "https://llm.test/v1/chat/completions"))
    return openai.BadRequestError(message, response=response, body={"error": {"code": "tool_use_failed"}})


@pytest.fixture
def make_orchestrator(training_stubs, settings):
    def _make(replies: list[AIMessage], **kwargs) -> tuple[Orchestrator, StubLLM]:
        llm = StubLLM(replies)
        return Orchestrator(settings, llm=llm, **kwargs), llm

    return _make


def tool_messages(orchestrator: Orchestrator) -> list[ToolMessage]:
    return [message for message in orchestrator.chat_history if isinstance(message, ToolMessage)]


def test_tools_are_bound_without_parallel_calls(make_orchestrator):
    _, llm = make_orchestrator([])

    assert llm.bind_kwargs == {"parallel_tool_calls": False}
    assert "run_experiment" in llm.bound_tool_names
    assert "finish" in llm.bound_tool_names


def test_tool_result_is_fed_back_to_the_llm(make_orchestrator):
    orchestrator, llm = make_orchestrator(
        [reply(call("check_budget_status")), reply(call("finish", {"reason": "done"}, "call_2"))],
        max_experiments=4,
    )

    orchestrator.run()

    assert len(llm.seen) == 2
    fed_back = llm.seen[1][-1]
    assert isinstance(fed_back, ToolMessage)
    assert fed_back.tool_call_id == "call_1"
    assert "4" in fed_back.content


def test_run_starts_from_the_system_prompt_and_a_task(make_orchestrator):
    orchestrator, llm = make_orchestrator([reply(call("finish", {"reason": "done"}))])

    orchestrator.run()

    assert isinstance(llm.seen[0][0], SystemMessage)
    assert isinstance(llm.seen[0][1], HumanMessage)


def test_stops_when_the_agent_finishes(make_orchestrator, training_stubs):
    orchestrator, llm = make_orchestrator(
        [
            reply(call("run_experiment", EXPERIMENT_ARGS)),
            reply(call("finish", {"reason": "good enough"}, "call_2")),
            reply(call("run_experiment", EXPERIMENT_ARGS, "call_3")),
        ]
    )

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert result.steps == 2
    assert result.experiments == 1
    assert len(training_stubs.fine_tune_calls) == 1
    assert [message.tool_call_id for message in tool_messages(orchestrator)] == ["call_1", "call_2"]


def test_stops_when_the_budget_is_exhausted(make_orchestrator, training_stubs):
    orchestrator, llm = make_orchestrator(
        [reply(call("run_experiment", EXPERIMENT_ARGS, f"call_{i}")) for i in range(5)],
        max_experiments=2,
    )

    result = orchestrator.run()

    assert result.stop_reason == "budget_exhausted"
    assert result.experiments == 2
    assert len(training_stubs.fine_tune_calls) == 2
    assert len(llm.seen) == 2


def test_stops_at_the_step_limit(make_orchestrator):
    orchestrator, llm = make_orchestrator(
        [reply(call("check_budget_status", call_id=f"call_{i}")) for i in range(10)],
        max_steps=3,
    )

    result = orchestrator.run()

    assert result.stop_reason == "max_steps"
    assert result.steps == 3
    assert len(llm.seen) == 3


def test_parallel_tool_calls_run_only_the_first(make_orchestrator, training_stubs):
    orchestrator, _ = make_orchestrator(
        [
            reply(
                call("run_experiment", EXPERIMENT_ARGS, "call_a"),
                call("run_experiment", {**EXPERIMENT_ARGS, "model_name": "swin_s"}, "call_b"),
                call("check_budget_status", call_id="call_c"),
            ),
            reply(call("finish", {"reason": "done"}, "call_d")),
        ]
    )

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert training_stubs.built_models == ["resnet50"]
    answers = {message.tool_call_id: message.content for message in tool_messages(orchestrator)}
    assert set(answers) == {"call_a", "call_b", "call_c", "call_d"}
    assert not answers["call_a"].startswith("ERROR")
    assert answers["call_b"].startswith("ERROR")
    assert answers["call_c"].startswith("ERROR")


def test_finish_is_ignored_when_it_is_not_the_first_call(make_orchestrator):
    orchestrator, _ = make_orchestrator(
        [reply(call("check_budget_status", call_id="call_a"), call("finish", {"reason": "done"}, "call_b"))],
        max_steps=1,
    )

    result = orchestrator.run()

    assert result.stop_reason == "max_steps"


def test_tool_exception_is_returned_to_the_llm(make_orchestrator, training_stubs):
    training_stubs.fine_tune_error = RuntimeError("CUDA out of memory")
    orchestrator, llm = make_orchestrator(
        [reply(call("run_experiment", EXPERIMENT_ARGS)), reply(call("finish", {"reason": "done"}, "call_2"))]
    )

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert result.experiments == 0
    error = llm.seen[1][-1]
    assert isinstance(error, ToolMessage)
    assert error.content.startswith("ERROR")
    assert "CUDA out of memory" in error.content


def test_invalid_arguments_are_returned_to_the_llm(make_orchestrator, training_stubs):
    orchestrator, llm = make_orchestrator(
        [
            reply(call("run_experiment", {**EXPERIMENT_ARGS, "epochs": 0})),
            reply(call("finish", {"reason": "done"}, "call_2")),
        ]
    )

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert training_stubs.fine_tune_calls == []
    assert llm.seen[1][-1].content.startswith("ERROR")
    assert "epochs" in llm.seen[1][-1].content


def test_unknown_tool_is_returned_to_the_llm(make_orchestrator):
    orchestrator, llm = make_orchestrator(
        [reply(call("get_experience")), reply(call("finish", {"reason": "done"}, "call_2"))]
    )

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert llm.seen[1][-1].content.startswith("ERROR")
    assert "get_experience_summary" in llm.seen[1][-1].content


def test_reply_without_a_tool_call_is_nudged(make_orchestrator):
    orchestrator, llm = make_orchestrator(
        [reply(content="Let me think about it."), reply(call("finish", {"reason": "done"}))]
    )

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert isinstance(llm.seen[1][-2], AIMessage)
    assert isinstance(llm.seen[1][-1], HumanMessage)
    assert "finish" in llm.seen[1][-1].content


def test_llm_that_never_calls_a_tool_hits_the_step_limit(make_orchestrator):
    orchestrator, llm = make_orchestrator([], max_steps=4)

    result = orchestrator.run()

    assert result.stop_reason == "max_steps"
    assert len(llm.seen) == 4


def test_rejected_tool_call_is_retried(make_orchestrator):
    orchestrator, llm = make_orchestrator([rejected_tool_call(), reply(call("finish", {"reason": "done"}))])

    result = orchestrator.run()

    assert result.stop_reason == "finished"
    assert result.steps == 2
    nudge = llm.seen[1][-1]
    assert isinstance(nudge, HumanMessage)
    assert "run_experiment<|channel|>commentary" in nudge.content
    assert "run_experiment" in nudge.content and "finish" in nudge.content


def test_repeated_llm_errors_stop_the_run(make_orchestrator):
    orchestrator, llm = make_orchestrator([rejected_tool_call() for _ in range(10)])

    result = orchestrator.run()

    assert result.stop_reason == "llm_error"
    assert result.steps == 3
    assert len(llm.seen) == 3


def test_llm_error_count_resets_after_a_good_reply(make_orchestrator):
    orchestrator, _ = make_orchestrator(
        [
            rejected_tool_call(),
            rejected_tool_call(),
            reply(call("check_budget_status")),
            rejected_tool_call(),
            rejected_tool_call(),
            reply(call("finish", {"reason": "done"}, "call_2")),
        ]
    )

    result = orchestrator.run()

    assert result.stop_reason == "finished"


def test_approval_is_asked_only_for_experiments(make_orchestrator, training_stubs):
    asked = []

    def approve(args: dict) -> bool:
        asked.append(args)
        return True

    orchestrator, _ = make_orchestrator(
        [
            reply(call("check_budget_status", call_id="call_1")),
            reply(call("run_experiment", EXPERIMENT_ARGS, "call_2")),
            reply(call("finish", {"reason": "done"}, "call_3")),
        ],
        approve=approve,
    )

    result = orchestrator.run()

    assert asked == [EXPERIMENT_ARGS]
    assert result.experiments == 1
    assert result.stop_reason == "finished"


def test_declined_approval_stops_the_run(make_orchestrator, training_stubs):
    orchestrator, llm = make_orchestrator(
        [reply(call("run_experiment", EXPERIMENT_ARGS)), reply(call("run_experiment", EXPERIMENT_ARGS, "call_2"))],
        approve=lambda args: False,
    )

    result = orchestrator.run()

    assert result.stop_reason == "stopped_by_user"
    assert training_stubs.fine_tune_calls == []
    assert len(llm.seen) == 1
    assert [message.tool_call_id for message in tool_messages(orchestrator)] == ["call_1"]


def test_system_prompt_names_real_tools_and_the_real_budget(make_orchestrator):
    orchestrator, llm = make_orchestrator([], max_experiments=3)
    prompt = orchestrator.chat_history[0].content

    assert "Maximum 3 experiments" in prompt
    assert re.search(r"get_experience(?!_summary)", prompt) is None
    for name in re.findall(r"\b(?:get|check|run)_[a-z_]+\b", prompt):
        assert name in llm.bound_tool_names
    assert "finish" in prompt
    assert "I will prompt you" not in prompt


def test_system_prompt_targets_macro_f1_and_asks_for_a_rationale(make_orchestrator):
    orchestrator, _ = make_orchestrator([])
    prompt = " ".join(orchestrator.chat_history[0].content.split())

    assert "maximize validation macro-F1" in prompt
    assert "rationale" in prompt
    assert "validation accuracy improves" not in prompt


def test_system_prompt_warns_that_small_differences_are_noise(make_orchestrator):
    orchestrator, _ = make_orchestrator([])
    prompt = " ".join(orchestrator.chat_history[0].content.split())

    assert "more than 0.02" not in prompt
    assert "noise" in prompt


def test_real_llm_needs_an_api_key(settings):
    with pytest.raises(ConfigError, match="API_KEY"):
        Orchestrator(settings)
