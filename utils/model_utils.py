
import collections
import copy
from typing import List
import torch

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
