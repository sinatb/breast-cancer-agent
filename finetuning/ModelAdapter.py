from abc import ABC, abstractmethod

from torch import nn


# This method is used to return only the layers that are dense and not part of the classifier head.
# It is used to determine which layers should be targeted for LoRA adaptation.
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
    def replace_head(self, model: nn.Module, num_classes: int = 5) -> nn.Module:
        model.reset_classifier(num_classes)
        return model

    def get_classifier(self, model: nn.Module) -> nn.Module:
        return model.get_classifier()

    @abstractmethod
    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        pass


class VitAdapter(ModelAdapter):
    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]


class SwinAdapter(ModelAdapter):
    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2", "downsample.reduction"]


class ResNetAdapter(ModelAdapter):
    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return _dense_layer_names(model, self.get_classifier(model), skip=frozenset({"conv1"}))


class DenseNetAdapter(ModelAdapter):
    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return _dense_layer_names(model, self.get_classifier(model), skip=frozenset({"features.conv0"}))


class MobileNetV3Adapter(ModelAdapter):
    def get_lora_target_modules(self, model: nn.Module) -> list[str]:
        return _dense_layer_names(model, self.get_classifier(model), skip=frozenset({"conv_stem"}))
