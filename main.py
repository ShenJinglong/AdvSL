
import copy
import wandb
import torch
import torchvision
import logging

from ClusterUnion import ClusterUnion
from model.resnet import ResNet18_Mnist
from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, eval_model, ratio_model_grad
from utils.hardware_utils import get_free_gpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
wandb.init(
    project="GanSFL",
    entity="sjinglong"
) # 我们使用wandb对仿真的参数和数据进行管理，并进行可视化
config = wandb.config
DEVICE = f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu"
if config.add_gan:
    alg = "GanSFL"
    logging.info(f"@@ GanSFL [{DEVICE}]")
else:
    alg = "SFL"
    logging.info(f"@@ SFL [{DEVICE}]")

# 准备数据集
dataset_manager = DatasetManager("./data/", config.percent, config.batch_size) # 参数中的 percent 指定用数据集中的百分之多少进行训练
datasets = dataset_manager.datasets
# datasets = ["MNIST", "MNIST_M"]
num_client = len(datasets)
trainloaders = dataset_manager.get_trainloaders(datasets)
testloaders = dataset_manager.get_testloaders(datasets)

# 准备模型
global_model = ResNet18_Mnist().to(DEVICE)
client_globalmodel = global_model.get_splited_module(config.cut_point)[0] # client 侧的全局模型
server_globalmodel = global_model.get_splited_module(config.cut_point)[1] # server 侧的全局模型
client_localmodels = [copy.deepcopy(client_globalmodel) for _ in range(num_client)] # client 侧的本地模型
with torch.no_grad():
    fm_size = client_localmodels[0](torch.randn((1, 3, 28, 28), device=DEVICE)).size() # 获取切割层输出的 feature map 的 size

# 优化器，loss等
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE) # 正常 SL 训练的 loss
server_optim = torch.optim.SGD(server_globalmodel.parameters(), config.lr) # 正常 SL 训练的 optim
client_optims = [torch.optim.SGD(model.parameters(), config.lr) for model in client_localmodels]

# 鉴别器
if config.add_gan:
    cluster_unions = [ClusterUnion(config.k, fm_size, config.lr, DEVICE) for _ in range(num_client-1)]

if __name__ == "__main__":
    for round in range(config.global_round):
        logging.info(f"round: {round}")

        # 下发模型
        [model.load_state_dict(client_globalmodel.state_dict()) for model in client_localmodels] # 将 client globalmodel 加载到 client localmodels 里边

        # training
        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")
            for one_batch_datas in zip(*trainloaders): # 从各个用户的数据集中读一个 batch 的数据，放在列表里
                samples = [one_batch_data[0].to(DEVICE) for one_batch_data in one_batch_datas] # 输入样本
                labels = [one_batch_data[1].to(DEVICE, dtype=torch.int64) for one_batch_data in one_batch_datas]  # 对应标签

                [client_optim.zero_grad() for client_optim in client_optims]
                server_optim.zero_grad()
                
                feature_maps = [client_localmodel(sample) for client_localmodel, sample in zip(client_localmodels, samples)] # 各个 client 输出 feature map

############################################################ 鉴别器起作用的部分， 去掉就是单纯的SL ######################################################################
#######################################################################################################################################################################
                if config.add_gan:
                    # cluster_union.update(feature_maps[config.target_domain], feature_maps[:config.target_domain] + feature_maps[config.target_domain+1:])
                    [cluster_union.update_gr_(feature_maps[config.target_domain], [feature_map]) for cluster_union, feature_map in zip(cluster_unions, feature_maps[:config.target_domain] + feature_maps[config.target_domain+1:])]
                    # 对鉴别器反向的梯度进行加权
                    [ratio_model_grad(model, config.ratio_gan) if i != config.target_domain else None for i, model in enumerate(client_localmodels)]
                    # [ratio_model_grad(model, config.ratio_gan) for model in client_localmodels]
#######################################################################################################################################################################

                # 将 feature map 输入 server model，并反向传播，得到 SL 的梯度，这些梯度会与之前 gan 的加权梯度加起来
                outputs = [server_globalmodel(feature_map) for feature_map in feature_maps]
                losses = [loss_fn(output, label) for output, label in zip(outputs, labels)]
                [loss.backward() for loss in losses]
                ratio_model_grad(server_globalmodel, 1/num_client)

                # 更新 localmodels
                [client_optim.step() for client_optim in client_optims]
                server_optim.step()
        # 将各个用户最后一个 batch 的第一个样本计算得到的 feature map 保存起来，用来观察 gan 对 feature map 的影响
        # 图像保存在 images 文件夹下，每个 round 保存一张
        # 图像中，每一行表示对应用户输出的各层 feature map；每一列表示 feature map 对应 channel 在不同用户上的区别
        torchvision.utils.save_image(torch.concat([feature_map[0].reshape(fm_size[1], 1, fm_size[2], fm_size[3]) for feature_map in feature_maps], dim=0), f"images/{alg}/{round}.png", nrow=fm_size[1], normalize=True)

        # 聚合模型
        client_globalmodel = aggregate_model(client_localmodels, [1 / num_client] * num_client)
        # 评估模型精度
        accs = [eval_model(client_globalmodel, server_globalmodel, testloader) for testloader in testloaders]
        # 日志
        logging_info = "acc:"
        for i, acc in enumerate(accs):
            logging_info += f" {datasets[i]}[{acc:.4f}] |"
        total_acc = sum(accs)
        logging_info += f" TOTAL_ACC[{total_acc:.4f}]"
        logging.info(logging_info)
        wandb.log({
            "round": round,
            "total_acc": total_acc,
            **{f"{dataset} acc": acc for dataset, acc in zip(datasets, accs)}
        })