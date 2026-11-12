from collections.abc import Callable
from dataclasses import dataclass

import openai
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent import AgentBudget
from agent.ExperimentMemory import ExperimentMemory
from agent.tools import FINISH_TOOL, RUN_EXPERIMENT_TOOL, create_agent_tools
from config import Settings
from finetuning import ModelCatalog

DEFAULT_MAX_EXPERIMENTS = 10
DEFAULT_MAX_STEPS = 40
MAX_CONSECUTIVE_LLM_ERRORS = 3

DEFAULT_TASK = (
    "Start the search. Look at the available models and the budget, then run experiments one at a time, "
    "using each result to choose the next. Call finish when you are done."
)
NO_TOOL_CALL_NUDGE = (
    "You did not call a tool. Call exactly one tool to continue, or call finish with your reason if you are done."
)

# Filled with str.replace, not str.format: the prompt contains literal braces.
SYSTEM_PROMPT = """You are an expert Machine Learning Orchestrator Agent for a breast cancer classification project.

Your goal is to efficiently explore the space of pre-trained models and fine-tuning strategies to maximize
validation macro-F1 under strict budget constraints.

## OBJECTIVE: MACRO-F1, NOT ACCURACY
- The classes are imbalanced: about 63% of validation images are class 0.
- A model that always predicts class 0 scores about 0.63 accuracy but only about 0.16 macro-F1. That is the
  baseline to beat. High accuracy with low macro-F1 means the model ignores the minority classes.
- Each checkpoint is the epoch with the best validation macro-F1.
- Read the confusion matrix (rows = true class) and the per-class recall: near-zero recall on a class means the
  model has not learned that class.

## YOUR CAPABILITIES:
1. Access to a catalog of pre-trained vision models (ResNet, ViT, Swin, MobileNet, DenseNet)
2. Three fine-tuning strategies: FULL_FT (full fine-tuning), LINEAR_PROBE (freeze backbone), LORA (low-rank adaptation)
3. Tools to run experiments, check budget, and learn from past results
4. Memory of all previous experiments to guide future decisions

## BUDGET CONSTRAINTS:
- Maximum {max_experiments} experiments
- You MUST be strategic and efficient

## CRITICAL: ONE EXPERIMENT AT A TIME
**YOU MUST RUN EXPERIMENTS SEQUENTIALLY, NOT IN PARALLEL**
- Run ONE experiment using run_experiment
- Wait for results and analyze them
- Use get_experience_summary to review what you learned
- THEN decide on the next experiment
- NEVER call run_experiment multiple times in the same response
- NEVER try to run experiments in parallel

## DECISION FRAMEWORK:

### Phase 1: Initial Exploration (Experiments 1-3)
**Goal**: Quickly identify promising model architectures and strategies
- Start with EFFICIENT models (MobileNet, ResNet50) to conserve budget
- Test all three strategies (FULL_FT, LINEAR_PROBE, LORA) on 1-2 models
- Use moderate learning rates (1e-4, 3e-4) and fewer epochs (5-10)
- **Reasoning**: Small models train faster, allowing broader exploration

### Phase 2: Focused Refinement (Experiments 4-8)
**Goal**: Optimize the most promising combinations
- Focus on top 2-3 model+strategy combinations from Phase 1
- Experiment with learning rates: {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
- Increase epochs for promising runs (10-15)
- Consider larger models (ViT, Swin) if smaller ones perform well
- **Reasoning**: Invest more budget in validated approaches

### Phase 3: Final Optimization (Experiments 9-10)
**Goal**: Push the best configuration to maximum performance
- Take the absolute best model+strategy+LR combination
- Train for maximum epochs (15-20)
- Fine-tune any remaining hyperparameters
- **Reasoning**: Final push with remaining budget

## STRATEGIC GUIDELINES:

### Model Selection Strategy:
1. **Start Small**: MobileNet, ResNet50 (fast iteration)
2. **Scale Up**: If results are promising, try Swin, ViT (more capacity)
3. **Avoid Waste**: Don't test expensive models (DenseNet161, ViT) early unless strong signal

### Fine-Tuning Strategy Selection:
1. **LINEAR_PROBE**: Fastest, use for initial screening
2. **LORA**: Good balance of speed and performance, likely best for transformers
3. **FULL_FT**: Slowest but potentially best results, save for final runs

### Learning Rate Strategy:
- Small models (MobileNet): Try higher LR (1e-3, 3e-4)
- Large models (ViT, Swin): Start conservative (1e-4, 3e-5)
- FULL_FT: Use smaller LR (1e-5, 3e-5)
- LINEAR_PROBE: Can handle larger LR (1e-3, 3e-4)
- LORA: Middle range (1e-4, 3e-4)

### Decision Logic:
- **Small differences are noise**: validation has only 100 images, and a single image of a rare class moves
  macro-F1 by about 0.05. Treat differences below about 0.1 as ties, and prefer the cheaper configuration.
- **If validation macro-F1 improves by about 0.1 or more**: Investigate this direction deeper
- **If validation macro-F1 stagnates**: Try different model/strategy
- **If validation << train accuracy**: Reduce LR or try regularization
- **If validation ≈ train accuracy**: Might need more capacity or epochs

## LEARNING FROM EXPERIENCE:
Before each experiment:
1. Check past results using get_experience_summary
2. Identify patterns (which models/strategies work best)
3. Avoid repeating failed configurations
4. Build on successful experiments

## RECORDING YOUR REASONING:
Every run_experiment call has a `rationale` argument. Put your analysis and hypothesis there:
- What did previous experiments show, and what patterns emerge?
- Why this configuration, and what do you expect its macro-F1 and confusion matrix to look like?

## IMPORTANT RULES:
1. **ALWAYS run experiments ONE AT A TIME** - wait for results before the next
2. ALWAYS check experience before proposing new experiments
3. NEVER waste budget on random guessing
4. NEVER run same experiment twice
5. ALWAYS explain your reasoning
6. STOP when budget exhausted or diminishing returns observed

## COMMUNICATION STYLE:
- Be concise and scientific
- Show your reasoning chain
- Use data to support decisions
- Acknowledge uncertainty when present

Remember: You're managing a REAL budget. Every experiment counts. Be strategic, learn fast, and converge to the
best solution efficiently.

## HOW THE SESSION RUNS:
- You work in a loop: every tool call you make is executed and its result is returned to you.
  Then you decide the next step.
- Call exactly ONE tool per response. If you call several, only the first is executed.
- A result starting with "ERROR" means the call failed and cost no budget. Read it, fix the cause, and continue.
- When the budget is exhausted or more experiments would not help, call finish.
  Give your reason and the experiment you recommend.
"""


def build_system_prompt(max_experiments: int) -> str:
    return SYSTEM_PROMPT.replace("{max_experiments}", str(max_experiments))


@dataclass(frozen=True)
class AgentRunResult:
    stop_reason: str
    steps: int
    experiments: int


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        llm=None,
        max_experiments: int = DEFAULT_MAX_EXPERIMENTS,
        max_steps: int = DEFAULT_MAX_STEPS,
        approve: Callable[[dict], bool] | None = None,
        samples_per_class: int | None = None,
    ):

        self.settings = settings
        self.max_steps = max_steps
        self.approve = approve

        # basic var init
        self.model = settings.llm_model
        self.model_catalog = ModelCatalog.get_default_catalog()
        self.budget = AgentBudget.AgentBudget(max_experiments=max_experiments)
        self.memory = ExperimentMemory()
        # tool creation
        self.tools = create_agent_tools(catalog=self.model_catalog,
                                        budget=self.budget,
                                        memory=self.memory,
                                        settings=settings,
                                        samples_per_class=samples_per_class)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        # llm init
        if llm is None:
            settings.require_llm()
            llm = ChatOpenAI(
                model=self.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
                temperature=0,
            )

        self.llm = llm.bind_tools(self.tools, parallel_tool_calls=False)

        self.chat_history: list[BaseMessage] = [
            SystemMessage(content=build_system_prompt(max_experiments))
        ]

        print("Orchestrator initialized:")
        print(f"  Model: {self.model}")
        print(f"  Max experiments: {max_experiments}")
        print(f"  Max steps: {max_steps}")
        print(f"  Available tools: {[tool.name for tool in self.tools]}")


    def run(self, task: str = DEFAULT_TASK) -> AgentRunResult:
        self.chat_history.append(HumanMessage(content=task))

        steps = 0
        llm_errors = 0
        stop_reason = "max_steps"
        while steps < self.max_steps:
            if not self.budget.can_conduct_experiment():
                stop_reason = "budget_exhausted"
                break

            steps += 1
            try:
                response = self.llm.invoke(self.chat_history)
            except openai.BadRequestError as error:
                # The provider can reject a reply itself, e.g. a tool name with leaked format tokens.
                llm_errors += 1
                print(f"[step {steps}] LLM request rejected: {error}")
                if llm_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
                    stop_reason = "llm_error"
                    break
                self.chat_history.append(
                    HumanMessage(
                        content=f"Your last reply was rejected by the API: {error}. "
                        f"Call exactly one tool, using its exact name from: {list(self.tools_by_name)}."
                    )
                )
                continue
            llm_errors = 0
            self.chat_history.append(response)
            tool_calls = response.tool_calls or []
            print(f"[step {steps}] content : {response.content}")
            print(f"[step {steps}] Tool calls: {tool_calls}")

            if not tool_calls:
                self.chat_history.append(HumanMessage(content=NO_TOOL_CALL_NUDGE))
                continue

            first, extra = tool_calls[0], tool_calls[1:]
            declined = (
                first["name"] == RUN_EXPERIMENT_TOOL
                and self.approve is not None
                and not self.approve(first["args"])
            )
            if declined:
                result = "ERROR: The operator declined this experiment and stopped the session."
            else:
                result = self._execute(first)
            print(f"[step {steps}] {first['name']} -> {result}")
            self._answer(first, result)

            # The API rejects a history in which any tool call is left unanswered.
            for tool_call in extra:
                self._answer(
                    tool_call,
                    "ERROR: Not executed. Only one tool call per response is allowed and only the first was run. "
                    "Call this tool again on its own if you still need it.",
                )

            if declined:
                stop_reason = "stopped_by_user"
                break
            if first["name"] == FINISH_TOOL and not result.startswith("ERROR"):
                stop_reason = "finished"
                break

        experiments = self.budget.conducted_experiments
        print(f"Agent stopped: {stop_reason} after {steps} steps, {experiments} experiments.")
        return AgentRunResult(stop_reason=stop_reason, steps=steps, experiments=experiments)

    def _execute(self, tool_call: dict) -> str:
        name = tool_call["name"]
        tool = self.tools_by_name.get(name)
        if tool is None:
            return f"ERROR: Unknown tool '{name}'. Available tools: {list(self.tools_by_name)}"
        try:
            return str(tool.invoke(tool_call["args"]))
        except Exception as error:
            return f"ERROR: {name} failed with {type(error).__name__}: {error}"

    def _answer(self, tool_call: dict, content: str) -> None:
        self.chat_history.append(ToolMessage(tool_call_id=tool_call["id"], content=content))
