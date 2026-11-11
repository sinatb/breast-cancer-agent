from finetuning.FineTuneResult import FineTuneResult


class ExperimentMemory:
    def __init__(self):
        self.conducted_experiments = []

    def add_experiment(self, experiment:FineTuneResult):
        self.conducted_experiments.append(experiment)
    def get_summary(self)->str:
        summary = ""
        for experiment in self.conducted_experiments:
            summary += experiment.get_summary() + '\n'
        return summary
