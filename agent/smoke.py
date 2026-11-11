"""Smoke run: the real loop, tools and training, with a scripted stand-in for the LLM and a small image sample.

It checks that a checkout is wired up correctly (configuration, data, weights, training, checkpoints) without an
API key. The numbers it produces mean nothing.
"""

from langchain_core.messages import AIMessage

from agent.tools import FINISH_TOOL, RUN_EXPERIMENT_TOOL

# 16 images of each of the 5 classes: 64 train, 8 validation, 8 test.
SMOKE_SAMPLES_PER_CLASS = 16

# The two smallest downloads in the catalog, covering a CNN probe and LoRA.
SMOKE_EXPERIMENTS = [
    {
        "model_name": "mobilenet_v3_large",
        "strategy": "LINEAR_PROBE",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 2,
        "batch_size": 8,
    },
    {
        "model_name": "resnet50",
        "strategy": "LORA",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 2,
        "batch_size": 8,
    },
]


class ScriptedLLM:
    """Stands in for a chat model: replays a fixed list of tool calls, one per reply, then calls finish."""

    def __init__(self, tool_calls: list[dict]):
        self.tool_calls = list(tool_calls)
        self.replies = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages) -> AIMessage:
        if self.replies < len(self.tool_calls):
            tool_call = self.tool_calls[self.replies]
        else:
            tool_call = {"name": FINISH_TOOL, "args": {"reason": "The script is complete."}}
        self.replies += 1
        return AIMessage(
            content=f"Scripted step {self.replies}.",
            tool_calls=[{**tool_call, "id": f"scripted_{self.replies}"}],
        )


def smoke_llm() -> ScriptedLLM:
    return ScriptedLLM(
        [
            {"name": "get_available_models", "args": {}},
            {"name": "check_budget_status", "args": {}},
            *({"name": RUN_EXPERIMENT_TOOL, "args": experiment} for experiment in SMOKE_EXPERIMENTS),
            {"name": "get_experience_summary", "args": {}},
        ]
    )


def smoke_passed(result) -> bool:
    return result.stop_reason == "finished" and result.experiments == len(SMOKE_EXPERIMENTS)
