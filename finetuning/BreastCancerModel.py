import torchvision.models as models
from torch import nn
from ModelCatalog import ModelData

model_map = {
    "vit_b_16" : models.vit_b_16,
    "swin_s" : models.swin_s,
    "resnet50" : models.resnet50,
    "mobilenet_v3_large" : models.mobilenet_v3_large,
    "densenet161" : models.densenet161,
}


def create_base_model(model_data : ModelData):
    new_model = model_map[model_data.name](weights="IMAGENET1K_V1")
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

    def forward(self, x):
        return self.base_model(x)
