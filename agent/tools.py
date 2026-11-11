import warnings

from langchain.tools import tool
from pydantic import BaseModel, Field

from agent.AgentBudget import AgentBudget
from agent.ExperimentMemory import ExperimentMemory
from config import Settings
from dataset.DataLoader import create_dataloaders
from finetuning.FineTuner import fine_tune
from finetuning.FineTuneStrategy import FineTuneStrategy
from finetuning.ModelCatalog import ModelCatalog
from utils.model_utils import get_model

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")

## TOOL INPUT DEFINITION
class GetModelsInput(BaseModel):
    pass

class CheckBudgetInput(BaseModel):
    pass

class RunExperimentInput(BaseModel):
    model_name: str = Field(description="Name of the model from catalog ('resnet50', 'vit_b_16', 'swin_s', 'mobilenet_v3_large', 'densenet161')")
    strategy: str = Field(description="Fine-tuning strategy: 'FULL_FT', 'LINEAR_PROBE', or 'LORA'")
    learning_rate: float = Field(description="Learning rate (e.g., 1e-4, 3e-4, 1e-3)")
    weight_decay: float = Field(description="Weight decay (e.g., 1e-4, 3e-4, 1e-3)")
    epochs: int = Field(description="Number of epochs to train (max 20)")
    batch_size: int = Field(default=8, description="Batch size for training")

class GetExperienceInput(BaseModel):
    pass


def create_agent_tools(catalog:ModelCatalog, budget:AgentBudget, memory:ExperimentMemory, settings:Settings):
    @tool(args_schema=GetModelsInput, description="Returns the list of available models.")
    def get_available_models() -> str:
        return str(catalog)
    @tool(args_schema=CheckBudgetInput, description="Returns the remaining budget.")
    def check_budget_status() -> str:
        return f"Remaining experiments : {budget.max_experiments - budget.conducted_experiments}"
    @tool(args_schema=RunExperimentInput, description="Fine-Tunes a model with the given conditions and saves results.")
    def run_experiment(
            model_name: str,
            strategy: str,
            learning_rate: float,
            weight_decay: float,
            epochs: int,
            batch_size: int = 32,
    ) -> str:

        if model_name not in catalog.model_dict:
            return f"ERROR: Unknown model '{model_name}'. Use get_available_models first."

        strategy_map = {
            "FULL_FT": FineTuneStrategy.FULL_FT,
            "LINEAR_PROBE": FineTuneStrategy.LINEAR_PROBE,
            "LORA": FineTuneStrategy.LORA
        }

        if strategy not in strategy_map.keys():
            return f"ERROR: Invalid strategy '{strategy}'. Must be one of: {list(strategy_map.keys())}"

        if epochs > 20:
            return f"ERROR: Epochs {epochs} exceeds maximum {20}"

        model_data = catalog.get_model_data(model_name)
        model = get_model(model_data)
        train_loader, val_loader, test_loader = create_dataloaders(settings.dataset_path, batch_size=batch_size)

        if not budget.can_conduct_experiment():
            return "ERROR: Insufficient budget."

        result = fine_tune(
            model=model,
            strategy=strategy_map[strategy],
            learning_rate=learning_rate,
            num_epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            model_dir=settings.model_dir,
            weight_decay=weight_decay
        )

        # Update memory and budget
        memory.add_experiment(result)
        budget.conduct_experiment()

        # Return summary
        summary = (
            f"✓ Experiment  completed\n"
            f"Results:\n"
            f"  Model name: {model_name}\n"
            f"  Strategy: {strategy}\n"
            f"  Learning rate: {learning_rate}\n"
            f"  Epochs: {epochs}\n"
            f"  Val Acc:   {result.accuracy:.4f} | Loss: {result.loss:.4f}\n"
            f"  Val Recall: {result.recall:.4f}"
            f"  Val Precision: {result.precision:.4f}"
            f"  Val F1: {result.f1_macro:.4f}"
            f"  Val cf: {result.confusion_matrix}"
            f"Remaining budget : {budget.max_experiments - budget.conducted_experiments}\n"
        )

        return summary
    @tool(args_schema=GetExperienceInput, description="Returns the past experiences.")
    def get_experience_summary() -> str:
        return memory.get_summary()
    return [
        get_available_models,
        check_budget_status,
        run_experiment,
        get_experience_summary
    ]
