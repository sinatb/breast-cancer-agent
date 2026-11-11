# Agent Architecture
This document explains how the orchestrator works and what are the tools at the agents disposal.
## Tools
The agent has 5 tools in total. All of these tools have been created using the `@tool` decorator from `langchain`. each 
tool has a specific input schema which is defined by the respective input schema.
1. `get_available_models` : Returns the model catalog to the agent. The model catalog contains all the models available 
for finetuning. The structure of the output is available at `finetuning_analysis.ipynb`.
2. `check_budget_status` : Returns the budget available for the model. `AgentBudget` is a simple class containing
information about the budget, including number of conducted experiments and the maximum number of experiments.
3. `run_experiment` : Finetunes a model given the parameters. This method directly uses the `fine_tune` method from 
finetuning package. After conducting an experiment, The result is stored at `ExperimentMemory` and the budget is reduced.
Its schema only accepts catalog models, the three strategies and bounded numbers (1 to 20 epochs, a positive learning
rate, a batch size up to 64), and a failed experiment costs no budget.
4. `get_experience_summary` : Returns a summary of all experiments conducted and their results. This is stored as a 
`FineTuneResult` object from finetuning package.
5. `finish` : Ends the session. The agent calls it with its reason once more experiments would not help.
## Orchestrator
The model used for the orchestrator is gpt-4.1-mini, the provider of this model was [metis](https://www.metisai.ir/). The
orchestrator creates a simple `ChatOpenAI` instance with tools bound to it and a simple list that is the chat history.
It should be mentioned that the history is initialized with the system prompt which guides the model on what it should do.
The main method for interacting with the orchestrator is `run`. It is a loop: the llm is called, the tool it chose is
executed, the result is appended to the history, and the llm is called again. The loop ends when the agent calls
`finish`, when the budget is spent, or after a maximum number of steps. A tool that fails (an invalid argument, an
out-of-memory error) does not end the session; the error is returned to the llm as the tool result. The history is
written to `history.txt` inside `MODEL_DIR` when the session ends, including when it crashes.
## Challenges
The main challenge in creating the agent was the fact that the agent would call tools in parallel. This means that at 
the first prompt, The agent would decide to call 3 fine_tune tools at the same time. Handling this was not feasible with
my setup, so the agent runs one tool per step. The tools are bound with `parallel_tool_calls=False`, and if a provider
ignores that, only the first call of a response is executed and the others are answered with an error. The prompts
also mention the constraint to reduce the chance of multiple tool call. The agent runs unattended by default; with
`--approve` the user is asked before each experiment and can stop the session there.