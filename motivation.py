
import copy
import os
import wandb
import torch
import logging
import torchvision

from utils.data_utils import DigitsDataset
from utils.model_utils import construct_model, eval_model_with_mutlitest, ratio_model_grad
from utils.hardware_utils import get_free_gpu

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("wandb").setLevel(logging.INFO)

wandb.init(
    project="AdvSL",
    entity="advsl",
    config={
        "dataset_setting": "IID"
    }
)
config = wandb.config

# DEVICE = f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

num_clients = 2
MNIST_M_trainset = DigitsDataset(
    root_path=os.path.join(os.path.abspath(os.environ.get("ADVSL_DATASET_PATH")), "digits"),
    domain="MNIST_M",
    channels=3,
    percent=1,
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)
MNIST_M_testset = DigitsDataset(
    root_path=os.path.join(os.path.abspath(os.environ.get("ADVSL_DATASET_PATH")), "digits"),
    domain="MNIST_M",
    channels=3,
    percent=1,
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)
SVHN_trainset = DigitsDataset(
    root_path=os.path.join(os.path.abspath(os.environ.get("ADVSL_DATASET_PATH")), "digits"),
    domain="SVHN",
    channels=3,
    percent=1,
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)
SVHN_testset = DigitsDataset(
    root_path=os.path.join(os.path.abspath(os.environ.get("ADVSL_DATASET_PATH")), "digits"),
    domain="SVHN",
    channels=3,
    percent=1,
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

MNIST_M_trainset = torch.utils.data.random_split(MNIST_M_trainset, lengths=[1024, len(MNIST_M_trainset)-1024])[0]
MNIST_M_testset = torch.utils.data.random_split(MNIST_M_testset, lengths=[1024, len(MNIST_M_testset)-1024])[0]
SVHN_trainset = torch.utils.data.random_split(SVHN_trainset, lengths=[1024, len(SVHN_trainset)-1024])[0]
SVHN_testset = torch.utils.data.random_split(SVHN_testset, lengths=[1024, len(SVHN_testset)-1024])[0]

if config.dataset_setting == "IID":
    trainsets = torch.utils.data.random_split(torch.utils.data.ConcatDataset([MNIST_M_trainset, SVHN_trainset]), lengths=[1024, 1024])
    testsets = torch.utils.data.random_split(torch.utils.data.ConcatDataset([MNIST_M_testset, SVHN_testset]), lengths=[1024, 1024])
elif config.dataset_setting == "Non-IID":
    trainsets = [MNIST_M_trainset, SVHN_trainset]
    testsets = [MNIST_M_testset, SVHN_testset]
else:
    raise ValueError()

trainloaders = [torch.utils.data.DataLoader(
    trainset,
    batch_size = 32,
    shuffle = True,
    drop_last = True,
) for trainset in trainsets]

testloaders = [torch.utils.data.DataLoader(
    testset,
    batch_size = 32,
    shuffle = False,
    drop_last = False,
) for testset in testsets]

global_model = construct_model("lenet", "digits").to(DEVICE)
client_globalmodel = global_model.get_splited_module(7)[0]
server_globalmodel = global_model.get_splited_module(7)[1]
client_localmodels = [copy.deepcopy(client_globalmodel) for _ in range(2)]

loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
server_optim = torch.optim.SGD(server_globalmodel.parameters(), 0.1) # 正常 SL 训练的 optim
client_optims = [torch.optim.SGD(model.parameters(), 0.1) for model in client_localmodels]

if __name__ == "__main__":
    batch_counter = 0 # 控制 testing 频率的计数器

    for round in range(100):
        logging.info(f"round: {round}")

        # training
        for epoch in range(2):
            logging.info(f"epoch: {epoch}")
            for one_batch_datas in zip(*trainloaders): # 从各个用户的数据集中读一个 batch 的数据，放在列表里
                samples = [one_batch_data[0].to(DEVICE) for one_batch_data in one_batch_datas] # 输入样本
                labels = [one_batch_data[1].to(DEVICE, dtype=torch.int64) for one_batch_data in one_batch_datas]  # 对应标签

                [client_optim.zero_grad() for client_optim in client_optims]
                server_optim.zero_grad()

                feature_maps = [client_localmodel(sample) for client_localmodel, sample in zip(client_localmodels, samples)] # 各个 client 输出 feature map
                outputs = [server_globalmodel(feature_map) for feature_map in feature_maps]
                losses = [loss_fn(output, label) for output, label in zip(outputs, labels)]
                [loss.backward() for loss in losses]
                ratio_model_grad(server_globalmodel, 1/2)

                # 更新 localmodels
                [client_optim.step() for client_optim in client_optims]
                server_optim.step()

                # 10个batch，1次test
                batch_counter += 1
                if batch_counter % 50 == 0:
                    accs = [eval_model_with_mutlitest(client_localmodel, server_globalmodel, testloaders) for client_localmodel in client_localmodels]
                    # 日志
                    logging_info = f"(round {round}, batch {batch_counter}) acc:"
                    for i, acc in enumerate(accs):
                        logging_info += f" client{i}[{acc:.4f}] |"
                    total_acc = sum(accs)
                    logging_info += f" TOTAL_ACC[{total_acc:.4f}]"
                    logging.info(logging_info)
                    wandb.log({
                        "round": round,
                        "batch": batch_counter,
                        "total_acc": total_acc,
                        **{f"client{i} acc": acc for i, acc in enumerate(accs)}
                    })

    torch.save(server_globalmodel, f"./saved_models/motivation/{config.dataset_setting}_server_model.pth")
    torch.save(client_localmodels[0], f"./saved_models/motivation/{config.dataset_setting}_client0_model.pth")
    torch.save(client_localmodels[1], f"./saved_models/motivation/{config.dataset_setting}_client1_model.pth")
    

