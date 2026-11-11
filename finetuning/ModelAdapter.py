from abc import ABC, abstractmethod

from torch import nn


def _dense_layer_names(model: nn.Module, head: nn.Module, skip: frozenset[str] = frozenset()) -> list[str]:
    head_modules = {id(m) for m in head.modules()}
    return [
        name
        for name, module in model.named_modules()
        if name not in skip
        and id(module) not in head_modules
        and (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d) and module.groups == 1))
    ]


class ModelAdapter(ABC):
    @abstractmethod
    def replace_head(self, model: nn.Module, num_classes: int) -> nn.Module:
        """Abstract method to replace classification head"""
        pass

    @abstractmethod
    def get_classifier(self, model) -> nn.Module:
        pass

    @abstractmethod
    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        pass


class VitAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def replace_head(self, model: nn.Module, num_classes=5) -> nn.Module:
        model.reset_classifier(num_classes)
        return model

    def get_classifier(self, model) -> nn.Module:
        return model.get_classifier()

    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]


class SwinAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def replace_head(self, model: nn.Module, num_classes=5) -> nn.Module:
        model.reset_classifier(num_classes)
        return model

    def get_classifier(self, model) -> nn.Module:
        return model.get_classifier()

    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2", "downsample.reduction"]


class ResNetAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def replace_head(self, model: nn.Module, num_classes=5) -> nn.Module:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def get_classifier(self, model) -> nn.Module:
        return model.fc

    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return _dense_layer_names(model, self.get_classifier(model), skip=frozenset({"conv1"}))


class DenseNetAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def replace_head(self, model: nn.Module, num_classes=5) -> nn.Module:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    def get_classifier(self, model) -> nn.Module:
        return model.classifier

    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return _dense_layer_names(model, self.get_classifier(model), skip=frozenset({"features.conv0"}))


class MobileNetV3Adapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def replace_head(self, model: nn.Module, num_classes=5) -> nn.Module:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    def get_classifier(self, model) -> nn.Module:
        return model.classifier[-1]

    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return _dense_layer_names(model, self.get_classifier(model), skip=frozenset({"features.0.0"}))
