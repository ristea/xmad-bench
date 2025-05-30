from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import torch

from detection.septr import SeparableTr
from detection.ast_model import ASTModelLocal


def get_resnet18_model(config):
    from einops.layers.torch import Rearrange
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  # weights=ResNet50_Weights.DEFAULT
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Sequential(*[
        torch.nn.Linear(model.fc.in_features, 2),
    ])
    model = model.to(config['device'])
    return model

def get_resnet50_model(config):
    from einops.layers.torch import Rearrange
    model = resnet50(weights=ResNet50_Weights.DEFAULT)  # weights=ResNet50_Weights.DEFAULT
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Sequential(*[
        torch.nn.Linear(model.fc.in_features, 2),
    ])
    model = model.to(config['device'])
    return model

def get_septr_model(config):
    model = SeparableTr(down_sample_input=(4, 4))
    model = model.to(config['device'])
    return model

def get_ast_model(config):
    model = ASTModelLocal()
    model = model.to(config['device'])
    return model