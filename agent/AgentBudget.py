from dataclasses import dataclass
@dataclass
class AgentBudget:

    max_experiments : int = 10
    conducted_experiments : int = 0

    def can_conduct_experiment(self):
        if self.conducted_experiments >= self.max_experiments:
            return False
        return True
    def conduct_experiment(self):
        self.conducted_experiments += 1
