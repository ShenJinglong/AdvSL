
import torch
import torchvision

from model.resnet import ResNet18_Mnist
from model.gan import Discriminator
from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, eval_model

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CUT_POINT = 2
LR = 0.001
PERCENT = 1
BATCH_SIZE = 64
GLOBAL_ROUND = 100
LOCAL_EPOCH = 2

dataset_manager = DatasetManager("./data/", PERCENT, BATCH_SIZE)
trainloaders = dataset_manager.get_trainloaders(dataset_manager.datasets)
testloaders = dataset_manager.get_testloaders(dataset_manager.datasets)

global_model = ResNet18_Mnist().to(DEVICE)
server_globalmodel = global_model.get_splited_module(CUT_POINT)[1]
client_globalmodel = global_model.get_splited_module(CUT_POINT)[0]
local_models = [ResNet18_Mnist().to(DEVICE) for _ in range(len(trainloaders))]
server_localmodels = [model.get_splited_module(CUT_POINT)[1] for model in local_models]
client_localmodels = [model.get_splited_module(CUT_POINT)[0] for model in local_models]
discriminator = Discriminator(64*28*28)

gan_loss = torch.nn.BCELoss().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
server_optims = [torch.optim.SGD(model.parameters(), LR) for model in server_localmodels]
client_optims = [torch.optim.SGD(model.parameters(), LR) for model in client_localmodels]

if __name__ == "__main__":
    for round in range(GLOBAL_ROUND):
        print(f"round {round}")

        [model.load_state_dict(server_globalmodel.state_dict()) for model in server_localmodels]
        [model.load_state_dict(client_globalmodel.state_dict()) for model in client_localmodels]
        
        # for i, (server_localmodel, client_localmodel, server_optim, client_optim, trainloader) in\
        #     enumerate(zip(server_localmodels, client_localmodels, server_optims, client_optims, trainloaders)):
        #     print(f"client {i} - {dataset_manager.datasets[i]}")
        #     for epoch in range(LOCAL_EPOCH):
        #         for inputs, label in trainloader:
        #             inputs, label = inputs.to(DEVICE), label.to(DEVICE)
        #             client_optim.zero_grad()
        #             server_optim.zero_grad()
        #             feature_map = client_localmodel(inputs)
        #             outputs = server_localmodel(feature_map)
        #             loss = loss_fn(outputs, label)
        #             loss.backward()

        #             client_optim.step()
        #             server_optim.step()
        #     torchvision.utils.save_image(feature_map[0].reshape((64, 1, 28, 28)), f"images/{round}-{i}.png", nrows=8, normalize=True)

        for epoch in range(LOCAL_EPOCH):
            print(f"epoch {epoch}")
            for one_batch_datas in zip(*trainloaders):
                samples = [one_batch_data[0].to(DEVICE) for one_batch_data in one_batch_datas]
                labels = [one_batch_data[1].to(DEVICE) for one_batch_data in one_batch_datas]

                [client_optim.zero_grad() for client_optim in client_optims]
                [server_optim.zero_grad() for server_optim in server_optims]
                
                feature_maps = [client_localmodel(sample) for client_localmodel, sample in zip(client_localmodels, samples)]

                outputs = [server_localmodel(feature_map) for server_localmodel, feature_map in zip(server_localmodels, feature_maps)]
                losses = [loss_fn(output, label) for output, label in zip(outputs, labels)]
                [loss.backward() for loss in losses]
                
                [client_optim.step() for client_optim in client_optims]
                [server_optim.step() for server_optim in server_optims]

        client_globalmodel = aggregate_model(client_localmodels, [0.2, 0.2, 0.2, 0.2, 0.2])
        server_globalmodel = aggregate_model(server_localmodels, [0.2, 0.2, 0.2, 0.2, 0.2])
        accs = [eval_model(client_globalmodel, server_globalmodel, testloader) for testloader in testloaders]
        print(accs)
