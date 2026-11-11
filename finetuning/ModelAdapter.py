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
    @abstractmethod
    def get_lora_target_modules(self)->list[str]:
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
    def get_lora_target_modules(self) -> list[str]:
        return ["self_attention.in_proj_weight", "self_attention.out_proj", "mlp.0", "mlp.3",]

class SwinAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.head
    def get_lora_target_modules(self) -> list[str]:
        return ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

class ResNetAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.fc
    def get_lora_target_modules(self) -> list[str]:
        return ["conv1", "conv2", "conv3"]

class DenseNetAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.classifier
    def get_lora_target_modules(self) -> list[str]:
        return ["conv1", "conv2"]

class MobileNetV3Adapter(ModelAdapter):
    def __init__(self):
        super().__init__()
    def replace_head(self, model: nn.Module, num_classes = 5) -> nn.Module:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    def get_classifier(self, model) -> nn.Module:
        return model.classifier[-1]
    def get_lora_target_modules(self) -> list[str]:
        return ["block.0.0", "block.1.0", "block.2.0", "features.0.0"]
