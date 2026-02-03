from abc import ABC, abstractmethod
from torch import nn


class ModelAdapter(ABC):
    @abstractmethod
    def replace_head(self, model: nn.Module, num_classes: int) -> nn.Module:
        """Abstract method to replace classification head"""
        pass
    @abstractmethod
    def get_classifier(self, model) -> nn.Module:
        pass

class VitAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.heads

class SwinAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.head

class ResNetAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.fc


class DenseNetAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.classifier

class MobileNetV3Adapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.classifier[-1]
