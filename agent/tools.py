import warnings
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from agent.AgentBudget import AgentBudget
from agent.ExperimentMemory import ExperimentMemory
from config import Settings
from dataset.DataLoader import create_dataloaders
from finetuning.FineTuner import fine_tune
from finetuning.FineTuneStrategy import FineTuneStrategy
from finetuning.ModelCatalog import ModelCatalog
from utils.model_utils import get_model, release_memory

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")

RUN_EXPERIMENT_TOOL = "run_experiment"
FINISH_TOOL = "finish"

MAX_EPOCHS = 20
MAX_BATCH_SIZE = 64
DEFAULT_BATCH_SIZE = 32


## TOOL INPUT DEFINITION
class GetModelsInput(BaseModel):
    pass

class CheckBudgetInput(BaseModel):
    pass

class GetExperienceInput(BaseModel):
    pass

class FinishInput(BaseModel):
    reason: str = Field(description="Why the search is over, and which experiment you recommend.")


def _run_experiment_schema(catalog: ModelCatalog) -> type[BaseModel]:
    model_names = tuple(catalog.model_dict)
    strategy_names = tuple(strategy.name for strategy in FineTuneStrategy)

    class RunExperimentInput(BaseModel):
        model_name: Literal[model_names] = Field(description="Name of the model from the catalog")
        strategy: Literal[strategy_names] = Field(description="Fine-tuning strategy")
        learning_rate: float = Field(gt=0, le=1, description="Learning rate (e.g., 1e-4, 3e-4, 1e-3)")
        weight_decay: float = Field(ge=0, le=1, description="Weight decay (e.g., 1e-4, 3e-4, 1e-3)")
        epochs: int = Field(ge=1, le=MAX_EPOCHS, description=f"Number of epochs to train (1 to {MAX_EPOCHS})")
        batch_size: int = Field(
            default=DEFAULT_BATCH_SIZE,
            ge=1,
            le=MAX_BATCH_SIZE,
            description=f"Batch size for training (1 to {MAX_BATCH_SIZE})",
        )

    return RunExperimentInput


def create_agent_tools(catalog:ModelCatalog, budget:AgentBudget, memory:ExperimentMemory, settings:Settings):
    @tool(args_schema=GetModelsInput, description="Returns the list of available models.")
    def get_available_models() -> str:
        return str(catalog)
    @tool(args_schema=CheckBudgetInput, description="Returns the remaining budget.")
    def check_budget_status() -> str:
        return f"Remaining experiments : {budget.max_experiments - budget.conducted_experiments}"
    @tool(
        RUN_EXPERIMENT_TOOL,
        args_schema=_run_experiment_schema(catalog),
        description="Fine-Tunes a model with the given conditions and saves results.",
    )
    def run_experiment(
            model_name: str,
            strategy: str,
            learning_rate: float,
            weight_decay: float,
            epochs: int,
            batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> str:

        if not budget.can_conduct_experiment():
            return "ERROR: Insufficient budget."

        model = train_loader = val_loader = None
        try:
            model = get_model(catalog.get_model_data(model_name))
            train_loader, val_loader, _ = create_dataloaders(settings.dataset_path, batch_size=batch_size)

            result = fine_tune(
                model=model,
                strategy=FineTuneStrategy[strategy],
                learning_rate=learning_rate,
                num_epochs=epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                model_dir=settings.model_dir,
                weight_decay=weight_decay
            )
        finally:
            del model, train_loader, val_loader
            release_memory()

        # Update memory and budget
        memory.add_experiment(result)
        budget.conduct_experiment()

        # Return summary
        summary = (
            f"✓ Experiment completed\n"
            f"Results:\n"
            f"  Model name: {model_name}\n"
            f"  Strategy: {strategy}\n"
            f"  Learning rate: {learning_rate}\n"
            f"  Epochs: {epochs}\n"
            f"  Val Acc: {result.accuracy:.4f} | Loss: {result.loss:.4f}\n"
            f"  Val Recall: {result.recall:.4f}\n"
            f"  Val Precision: {result.precision:.4f}\n"
            f"  Val F1: {result.f1_macro:.4f}\n"
            f"  Val cf: {result.confusion_matrix}\n"
            f"Remaining budget : {budget.max_experiments - budget.conducted_experiments}\n"
        )

        return summary
    @tool(args_schema=GetExperienceInput, description="Returns the past experiences.")
    def get_experience_summary() -> str:
        return memory.get_summary()
    @tool(FINISH_TOOL, args_schema=FinishInput, description="Ends the session. Call it when the search is over.")
    def finish(reason: str) -> str:
        return f"Session finished: {reason}"
    return [
        get_available_models,
        check_budget_status,
        run_experiment,
        get_experience_summary,
        finish,
    ]
