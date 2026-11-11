from functools import partial

import timm
import torchvision.models as models
from torch import nn

from .ModelCatalog import ModelData

model_map = {
    "vit_b_16": partial(timm.create_model, "deit_base_patch16_224.fb_in1k", pretrained=True),
    "swin_s": partial(timm.create_model, "swin_small_patch4_window7_224.ms_in1k", pretrained=True),
    "resnet50": partial(models.resnet50, weights="IMAGENET1K_V1"),
    "mobilenet_v3_large": partial(models.mobilenet_v3_large, weights="IMAGENET1K_V1"),
    "densenet161": partial(models.densenet161, weights="IMAGENET1K_V1"),
}


def create_base_model(model_data: ModelData):
    new_model = model_map[model_data.name]()
    new_model = model_data.adapter.replace_head(new_model, num_classes=5)
    return new_model


class BreastCancerModel(nn.Module):
    def __init__(self,
                 model_data:ModelData,
                 freeze_backbone = False):
        super().__init__()
        self.model_name = model_data.name
        self.adapter = model_data.adapter
        self.base_model = create_base_model(model_data)
        self.frozen_backbone = freeze_backbone
        if freeze_backbone:
            self.freeze_backbone()

    def unfreeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.frozen_backbone = False

    def freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        classifier_head = self.adapter.get_classifier(self.base_model)
        for param in classifier_head.parameters():
            param.requires_grad = True
        self.frozen_backbone = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen_backbone:
            head_modules = set(self.adapter.get_classifier(self.base_model).modules())
            for module in self.base_model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm) and module not in head_modules:
                    module.eval()
        return self

    def forward(self, x):
        return self.base_model(x)
