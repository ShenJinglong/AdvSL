
import enum
import wandb
import torch
import torchvision
import logging

from model.resnet import ResNet18_Mnist
from model.gan import Discriminator
from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, eval_model, ratio_model_grad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
wandb.init(
    project="GanSFL",
    entity="sjinglong"
) # 我们使用wandb对仿真的参数和数据进行管理，并进行可视化
config = wandb.config
if config.add_gan:
    logging.info("@@ GanSFL")
else:
    logging.info("@@ SFL")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 准备数据集
dataset_manager = DatasetManager("./data/", config.percent, config.batch_size) # 参数中的 percent 指定用数据集中的百分之多少进行训练
trainloaders = dataset_manager.get_trainloaders(dataset_manager.datasets)
testloaders = dataset_manager.get_testloaders(dataset_manager.datasets)

# 准备模型
global_model = ResNet18_Mnist().to(DEVICE)
local_models = [ResNet18_Mnist().to(DEVICE) for _ in range(len(trainloaders))]
server_globalmodel = global_model.get_splited_module(config.cut_point)[1] # server 侧的全局模型
client_globalmodel = global_model.get_splited_module(config.cut_point)[0] # client 侧的全局模型
server_localmodels = [model.get_splited_module(config.cut_point)[1] for model in local_models] # server 侧对应各个 client 的本地模型
client_localmodels = [model.get_splited_module(config.cut_point)[0] for model in local_models] # client 侧的本地模型
with torch.no_grad():
    fm_size = client_localmodels[0](torch.randn((1, 3, 28, 28), device=DEVICE)).size() # 获取切割层输出的 feature map 的 size
if config.add_gan:
    discriminator = Discriminator(fm_size.numel()).to(DEVICE) # 鉴别器

# 优化器，loss等
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE) # 正常 SFL 训练的 loss
server_optims = [torch.optim.SGD(model.parameters(), config.lr) for model in server_localmodels] # 正常 SFL 训练的 optim
client_optims = [torch.optim.SGD(model.parameters(), config.lr) for model in client_localmodels]
if config.add_gan:
    gan_loss_fn = torch.nn.BCELoss().to(DEVICE) # 鉴别器的 loss
    gan_optim = torch.optim.SGD(discriminator.parameters(), config.lr) # 用于更新鉴别器的 optim

if __name__ == "__main__":
    for round in range(config.global_round):
        logging.info(f"round: {round}")

        # 下发模型
        [model.load_state_dict(server_globalmodel.state_dict()) for model in server_localmodels] # 将 server globalmodel 加载到 server localmodels 里边
        [model.load_state_dict(client_globalmodel.state_dict()) for model in client_localmodels] # 将 client globalmodel 加载到 client localmodels 里边

        # training
        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")
            for one_batch_datas in zip(*trainloaders): # 从各个用户的数据集中读一个 batch 的数据，放在列表里
                samples = [one_batch_data[0].to(DEVICE) for one_batch_data in one_batch_datas] # 输入样本
                labels = [one_batch_data[1].to(DEVICE) for one_batch_data in one_batch_datas]  # 对应标签

                [client_optim.zero_grad() for client_optim in client_optims]
                [server_optim.zero_grad() for server_optim in server_optims]
                
                feature_maps = [client_localmodel(sample) for client_localmodel, sample in zip(client_localmodels, samples)] # 各个 client 输出 feature map

                if config.add_gan:
                    # 设置训练鉴别器的标签，目标域的标签为 1， 非目标域的标签为 0
                    gan_labels = [torch.zeros((feature_map.size(0), 1), device=DEVICE) if i != config.target_domain else torch.ones((feature_map.size(0), 1), device=DEVICE) for i, feature_map in enumerate(feature_maps)]
                    for _ in range(config.k): # 对鉴别器更新 k 步
                        for i, (feature_map, gan_label) in enumerate(zip(feature_maps, gan_labels)):
                            if i != config.target_domain:
                                gan_optim.zero_grad()
                                # 计算 loss：将非目标域的 feature map 输入，计算 loss1；将目标域的 feature map 输入，计算 loss2；总 loss 为 (loss1 + loss2) * 0.5
                                fake_gan_loss = gan_loss_fn(discriminator(feature_map.detach()), gan_label)
                                real_gan_loss = gan_loss_fn(discriminator(feature_maps[config.target_domain].detach()), gan_labels[config.target_domain])
                                d_loss = (fake_gan_loss + real_gan_loss) / 2
                                d_loss.backward()
                                gan_optim.step()
                    # 从鉴别器反向传播梯度到 client localmodel 上
                    gan_losses = [gan_loss_fn(discriminator(feature_map), torch.ones((feature_map.size(0), 1), device=DEVICE)) if i != config.target_domain else None for i, feature_map in enumerate(feature_maps)]
                    [gan_loss.backward(retain_graph=True) if gan_loss else None for gan_loss in gan_losses]
                    # 对 gan 反向的梯度进行加权
                    [ratio_model_grad(model, config.ratio_gan) if i != config.target_domain else None for i, model in enumerate(client_localmodels)]

                # 将 feature map 输入 server localmodels，并反向传播，得到 SFL 的梯度，这些梯度会与之前 gan 的加权梯度加起来
                outputs = [server_localmodel(feature_map) for server_localmodel, feature_map in zip(server_localmodels, feature_maps)]
                losses = [loss_fn(output, label) for output, label in zip(outputs, labels)]
                [loss.backward() for loss in losses]                

                # 更新 localmodels
                [client_optim.step() for client_optim in client_optims]
                [server_optim.step() for server_optim in server_optims]
        # 将各个用户最后一个 batch 的第一个样本计算得到的 feature map 保存起来，用来观察 gan 对 feature map 的影响
        # 图像保存在images文件夹下，每个 round 保存一张
        # 图像中，每一行表示对应用户输出的各层 feature map；每一列表示 feature map 对应 channel 在不同用户上的区别
        torchvision.utils.save_image(torch.concat([feature_map[0].reshape(fm_size[1], 1, fm_size[2], fm_size[3]) for feature_map in feature_maps], dim=0), f"images/{round}.png", nrow=fm_size[1], normalize=True)

        # 聚合模型
        client_globalmodel = aggregate_model(client_localmodels, [0.2, 0.2, 0.2, 0.2, 0.2])
        server_globalmodel = aggregate_model(server_localmodels, [0.2, 0.2, 0.2, 0.2, 0.2])
        # 评估模型精度
        accs = [eval_model(client_globalmodel, server_globalmodel, testloader) for testloader in testloaders]
        logging.info(f"acc: {dataset_manager.datasets[0]}[{accs[0]:.4f}] | {dataset_manager.datasets[1]}[{accs[1]:.4f}] | {dataset_manager.datasets[2]}[{accs[2]:.4f}] | {dataset_manager.datasets[3]}[{accs[3]:.4f}] | {dataset_manager.datasets[4]}[{accs[4]:.4f}]")
        wandb.log({
            "round": round,
            f"{dataset_manager.datasets[0]} acc": accs[0],
            f"{dataset_manager.datasets[1]} acc": accs[1],
            f"{dataset_manager.datasets[2]} acc": accs[2],
            f"{dataset_manager.datasets[3]} acc": accs[3],
            f"{dataset_manager.datasets[4]} acc": accs[4]
        })