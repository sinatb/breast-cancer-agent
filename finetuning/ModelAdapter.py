from abc import ABC, abstractmethod
from torch import nn


class ModelAdapter(ABC):
    @abstractmethod
    def replace_head(self, model: nn.Module, num_classes: int):
        """Abstract method to replace classification head"""
        pass

class VitAdapter(ModelAdapter):
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.heads.head.in_features
        model.heads = nn.Linear(in_features, num_classes)
        return model

class SwinAdapter(ModelAdapter):
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model

class ResNetAdapter(ModelAdapter):
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

class DenseNetAdapter(ModelAdapter):
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

class MobileNetV3Adapter(ModelAdapter):
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
