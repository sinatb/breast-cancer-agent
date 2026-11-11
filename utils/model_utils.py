from finetuning.BreastCancerModel import BreastCancerModel
from finetuning.ModelCatalog import ModelData


def get_model(model_data: ModelData, freeze_backbone=False):
    return BreastCancerModel(model_data, freeze_backbone)
