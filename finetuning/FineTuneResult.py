from dataclasses import dataclass, asdict
@dataclass
class FineTuneResult:
    model_name : str
    fine_tune_strategy : str
    model_dir : str
    learning_rate : float
    weight_decay : float
    epochs : int
    loss : float = float('inf')
    accuracy : float = -float('inf')
    precision : float = -float('inf')
    recall : float = -float('inf')
    f1_micro : float = -float('inf')
    f1_macro : float = -float('inf')
    f1_weighted : float = -float('inf')
    confusion_matrix : list = None
    def get_summary(self)->str:
        return f"model + strat : {self.model_name} + {self.fine_tune_strategy}, learning_rate : {self.learning_rate}, weight_decay : {self.weight_decay}, epochs : {self.epochs}, accuracy : {self.accuracy:.4f}, loss : {self.loss:.4f}, precision : {self.precision}, recall : {self.recall}, f1 : {self.f1_macro}"
    def to_dict(self)->dict:
        return asdict(self)