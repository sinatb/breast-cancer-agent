# Agent Architecture
This document explains how the orchestrator works and what are the tools at the agents disposal.
## Tools
The agent has 4 tools in total. All of these tools have been created using the `@tool` decorator from `langchain`. each 
tool has a specific input schema which is defined by the respective input schema.
1. `get_available_models` : Returns the model catalog to the agent. The model catalog contains all the models available 
for finetuning. The structure of the output is available at `finetuning_analysis.ipynb`.
2. `check_budget_status` : Returns the budget available for the model. `AgentBudget` is a simple class containing
information about the budget, including number of conducted experiments and the maximum number of experiments.
3. `run_experiment` : Finetunes a model given the parameters. This method directly uses the `fine_tune` method from 
finetuning package. After conducting an experiment, The result is stored at `ExperimentMemory` and the budget is reduced.
4. `get_experience_summary` : Returns a summary of all experiments conducted and their results. This is stored as a 
`FineTuneResult` object from finetuning package.
## Orchestrator
The model used for the orchestrator is gpt-4.1-mini, the provider of this model was [metis](https://www.metisai.ir/). The
orchestrator creates a simple `ChatOpenAI` instance with tools bound to it and a simple list that is the chat history.
It should be mentioned that the history is initialized with the system prompt which guides the model on what it should do.
The main method for interacting with the orchestrator is `invoke`. This method gets an input message and creates an llm
response base on that message. Tool calling is handled in this method. After the end of each session all of the history is
stored in a text file `history.txt`.
## Challenges
The main challenge in creating the agent was the fact that the agent would call tools in parallel. This means that at 
the first prompt, The agent would decide to call 3 fine_tune tools at the same time. Handling this was not feasible with
my setup, so I had to change the way I interacted with the llm. The current model is an interactive model, at each stage
the result of llm's thought process is displayed to the user and the user is asked if it should continue the process or not.
Also, the `invoke` method has a hard constraint on tool usage. Each prompt can only use 1 tool else it is discarded. The prompts
also mention the constraint to reduce the chance of multiple tool call.