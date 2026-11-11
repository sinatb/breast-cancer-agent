import gc

import torch

from finetuning.BreastCancerModel import BreastCancerModel
from finetuning.ModelCatalog import ModelData


def get_model(model_data: ModelData, freeze_backbone=False):
    return BreastCancerModel(model_data, freeze_backbone)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
