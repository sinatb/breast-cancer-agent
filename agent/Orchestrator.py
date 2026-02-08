import os
from agent import AgentBudget
from openai import OpenAI
from dotenv import load_dotenv
from finetuning import ModelCatalog
from agent.ExperimentMemory import ExperimentMemory

class Orchestrator(object):
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv("BASE_URL")
        self.api_key = os.getenv("API_KEY")
        self.model = "gpt-4.1-nano"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model_catalog = ModelCatalog.get_default_catalog()
        self.budget = AgentBudget.AgentBudget()
        self.memory = ExperimentMemory()

Orchestrator()