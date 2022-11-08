
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.data_utils import DigitsDataset

plt.style.use("ggplot")

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

MNIST_M_testset = torch.utils.data.random_split(MNIST_M_testset, lengths=[1024, len(MNIST_M_testset)-1024])[0]
SVHN_testset = torch.utils.data.random_split(SVHN_testset, lengths=[1024, len(SVHN_testset)-1024])[0]

IID_testsets = torch.utils.data.random_split(torch.utils.data.ConcatDataset([MNIST_M_testset, SVHN_testset]), lengths=[1024, 1024])
NonIID_testsets = [MNIST_M_testset, SVHN_testset]

IID_testloaders = [torch.utils.data.DataLoader(
    testset,
    batch_size = 32,
    shuffle = False,
    drop_last = False,
) for testset in IID_testsets]

NonIID_testloaders = [torch.utils.data.DataLoader(
    testset,
    batch_size = 32,
    shuffle = False,
    drop_last = False,
) for testset in NonIID_testsets]

IID_client0_features = []
IID_client0_labels = []
client_model = torch.load("./saved_models/motivation/IID_client0_model.pth").to("cpu")
with torch.no_grad():
    for x, y in IID_testloaders[0]:
        IID_client0_features.append(client_model(x))
        IID_client0_labels.append(y)
IID_client0_features = torch.concat(IID_client0_features, dim=0).numpy()
IID_client0_labels = torch.concat(IID_client0_labels, dim=0).numpy()

IID_client1_features = []
IID_client1_labels = []
client_model = torch.load("./saved_models/motivation/IID_client1_model.pth").to("cpu")
with torch.no_grad():
    for x, y in IID_testloaders[1]:
        IID_client1_features.append(client_model(x))
        IID_client1_labels.append(y)
IID_client1_features = torch.concat(IID_client1_features, dim=0).numpy()
IID_client1_labels = torch.concat(IID_client1_labels, dim=0).numpy()

IID_client_features = TSNE(
    n_components = 2,
    learning_rate = "auto",
    init = "random",
    perplexity=50
).fit_transform(np.concatenate([IID_client0_features, IID_client1_features], axis=0))
IID_client_labels = np.concatenate([IID_client0_labels, IID_client1_labels], axis=0)

plt.figure(1)
plt.scatter(IID_client_features[:1024, 0], IID_client_features[:1024, 1], s=14, c=IID_client_labels[:1024], marker="^")
plt.scatter(IID_client_features[1024:, 0], IID_client_features[1024:, 1], s=14, c=IID_client_labels[1024:], marker="o")
plt.tight_layout()
plt.savefig("/home/sjinglong/Desktop/motivation_iid.pdf")

NonIID_client0_features = []
NonIID_client0_labels = []
client_model = torch.load("./saved_models/motivation/Non-IID_client0_model.pth").to("cpu")
with torch.no_grad():
    for x, y in NonIID_testloaders[0]:
        NonIID_client0_features.append(client_model(x))
        NonIID_client0_labels.append(y)
NonIID_client0_features = torch.concat(NonIID_client0_features, dim=0).numpy()
NonIID_client0_labels = torch.concat(NonIID_client0_labels, dim=0).numpy()

NonIID_client1_features = []
NonIID_client1_labels = []
client_model = torch.load("./saved_models/motivation/Non-IID_client1_model.pth").to("cpu")
with torch.no_grad():
    for x, y in IID_testloaders[1]:
        NonIID_client1_features.append(client_model(x))
        NonIID_client1_labels.append(y)
NonIID_client1_features = torch.concat(NonIID_client1_features, dim=0).numpy()
NonIID_client1_labels = torch.concat(NonIID_client1_labels, dim=0).numpy()

NonIID_client_features = TSNE(
    n_components = 2,
    learning_rate = "auto",
    init = "random",
    perplexity=50
).fit_transform(np.concatenate([NonIID_client0_features, NonIID_client1_features], axis=0))
NonIID_client_labels = np.concatenate([NonIID_client0_labels, NonIID_client1_labels], axis=0)

plt.figure(2)
plt.scatter(NonIID_client_features[:1024, 0], NonIID_client_features[:1024, 1], s=14, c=NonIID_client_labels[:1024], marker="^")
plt.scatter(NonIID_client_features[1024:, 0], NonIID_client_features[1024:, 1], s=14, c=NonIID_client_labels[1024:], marker="o")
plt.tight_layout()
plt.savefig("/home/sjinglong/Desktop/motivation_noniid.pdf")
plt.show()

