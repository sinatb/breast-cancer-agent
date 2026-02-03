import torchvision.models as models
from torch import nn
from model_catalog import ModelData

model_map = {
    "vit_b_16" : models.vit_b_16,
    "swin_s" : models.swin_s,
    "resnet50" : models.resnet50,
    "mobilenet_v3_large" : models.mobilenet_v3_large,
    "densenet161" : models.densenet161,
}


def create_base_model(model : ModelData):
    new_model = model_map[model.name](weights="IMAGENET1K_V1")
    new_model = model.adapter.replace_head(new_model)
    return new_model

class BreastCancerModel(nn.Module):
    def __init__(self,
                 model:ModelData,
                 freeze_backbone = False):
        super().__init__()
        self.mode_name = model.name
        self.base_model = create_base_model(model)
        if freeze_backbone:
            self.__freeze_backbone()

    def __freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        classifier_head = self.__get_classifier()
        for param in classifier_head.parameters():
            param.requires_grad = True

    def __get_classifier(self):
        if self.model_name.startswith("vit"):
            return self.model.heads
        elif self.model_name.startswith("swin"):
            return self.model.head
        elif self.model_name.startswith("resnet"):
            return self.model.fc
        elif self.model_name.startswith("densenet"):
            return self.model.classifier
        elif self.model_name.startswith("mobilenet"):
            return self.model.classifier
    def forward(self, x):
        return self.base_model(x)

def get_model(model_data: ModelData, freeze_backbone = False):
    return BreastCancerModel(model_data, freeze_backbone)
