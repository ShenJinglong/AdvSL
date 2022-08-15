
import collections
import copy
from typing import List
import torch

from model.resnet import ResNet18_Mnist
from model.vgg import VGG16_Mnist
from model.alexnet import AlexNet_Mnist
from model.lenet import LeNet_Mnist

def construct_model(
    model_type: str,
) -> torch.nn.Module:
    if model_type == "vgg16":
        return VGG16_Mnist()
    elif model_type == "resnet18":
        return ResNet18_Mnist()
    elif model_type == "alexnet":
        return AlexNet_Mnist()
    elif model_type == "lenet":
        return LeNet_Mnist()
    else:
        raise ValueError(f"Unrecognized model type: `{model_type}`")

def aggregate_model(models: List[torch.nn.Module], weights: List[float]) -> torch.nn.Module:
    tmp_model = copy.deepcopy(models[0])

    global_dict = collections.OrderedDict()
    param_keys = tmp_model.state_dict().keys()
    for key in param_keys:
        sum = 0
        for i, model in enumerate(models):
            sum += weights[i] * model.state_dict()[key]
        global_dict[key] = sum

    tmp_model.load_state_dict(global_dict)
    return tmp_model

def eval_model(client_model: torch.nn.Module, server_model: torch.nn.Module, testloader: torch.utils.data.DataLoader) -> float:
    client_model.eval()
    server_model.eval()
    device = next(client_model.parameters()).device
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, label in testloader:
            inputs, label = inputs.to(device), label.to(device)
            fm = client_model(inputs)
            outputs = server_model(fm)
            _, pred = outputs.max(1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    client_model.train()
    server_model.train()
    return correct / total

def eval_model_with_mutlitest(client_model: torch.nn.Module, server_model: torch.nn.Module, testloaders: List[torch.utils.data.DataLoader]) -> float:
    client_model.eval()
    server_model.eval()
    device = next(client_model.parameters()).device
    total, correct = 0, 0
    with torch.no_grad():
        for testloader in testloaders:
            for inputs, label in testloader:
                inputs, label = inputs.to(device), label.to(device)
                fm = client_model(inputs)
                outputs = server_model(fm)
                _, pred = outputs.max(1)
                total += label.size(0)
                correct += (pred == label).sum().item()
    client_model.train()
    server_model.train()
    return correct / total

def ratio_model_grad(model: torch.nn.Module, ratio: float) -> None:
    for p in model.parameters():
        p.grad *= ratio

if __name__ == "__main__":
    model = torch.nn.Linear(1, 1, bias=False)
    input = torch.ones((1, 1))
    output = model(input)
    output.backward()
    for p in model.parameters():
        print(p.grad)
    ratio_model_grad(model, -0.6)
    for p in model.parameters():
        print(p.grad)
    input = 2*torch.ones((1, 1))
    output = model(input)
    output.backward()
    for p in model.parameters():
        print(p.grad)