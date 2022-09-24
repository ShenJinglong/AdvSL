
import copy
import wandb
import torch
import logging

from cluster import ClusterUnionAnchor, ClusterUnionMultiout
from utils.data_utils import DatasetManager, seed_torch
from utils.model_utils import aggregate_model, eval_model, ratio_model_grad, eval_model_with_mutlitest, construct_model
from utils.hardware_utils import get_free_gpu

# 设置随机数种子
# seed_torch(31)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
wandb.init(
    project="AdvSL",
    entity="advsl"
) # 我们使用wandb对仿真的参数和数据进行管理，并进行可视化
config = wandb.config
DEVICE = f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu" # 选择训练设备
if config.disc_type != "none":
    alg = "AdvSL"
    logging.info(f"@@ AdvSL [{DEVICE}]")
else:
    alg = "SL"
    logging.info(f"@@ SL [{DEVICE}]")

# 准备数据集
dataset_manager = DatasetManager("/home/sjinglong/datasets/personal/advsl/", config.percent, config.batch_size) # 参数中的 percent 指定用数据集中的百分之多少进行训练
# 设置训练使用的数据集，列表中数据集的数量也决定了参与训练的客户端的数量
datasets = [
    "MNIST",            # 原数据集
    "SVHN",
    "USPS",
    "SynthDigits",
    "MNIST_M",
    "MNIST-blur",       # 高斯模糊后的数据集（高斯核随机选择为1，3，5，7）
    "SVHN-blur",
    "USPS-blur",
    "SynthDigits-blur",
    "MNIST_M-blur",
    "MNIST-rot",        # 随机旋转后的数据集（随机进行小于60度的旋转）
    "SVHN-rot",
    "USPS-rot",
    "SynthDigits-rot",
    "MNIST_M-rot",
    "MNIST-noise",      # 添加了高斯白噪声后的数据集 (mean:0, std: 0.2)
    "SVHN-noise",
    "USPS-noise",
    "SynthDigits-noise",
    "MNIST_M-noise",
    "MNIST-bright",     # 随机调整了亮度的数据集
    "SVHN-bright",
    "USPS-bright",
    "SynthDigits-bright",
    "MNIST_M-bright",
    "MNIST-hue",        # 随机调整了色相的数据集
    "SVHN-hue",
    "USPS-hue",
    "SynthDigits-hue",
    "MNIST_M-hue",
]
num_client = len(datasets)
trainloaders = dataset_manager.get_trainloaders(datasets)
testloaders = dataset_manager.get_testloaders(datasets)
logging.debug(f"{num_client} clients with datasets {datasets}.")

# 准备模型
global_model = construct_model(config.model_type).to(DEVICE)
client_globalmodel = global_model.get_splited_module(config.cut_point)[0] # client 侧的全局模型
server_globalmodel = global_model.get_splited_module(config.cut_point)[1] # server 侧的全局模型
client_localmodels = [copy.deepcopy(client_globalmodel) for _ in range(num_client)] # client 侧的本地模型
with torch.no_grad():
    fm_size = client_localmodels[0](torch.randn((1, 3, 28, 28), device=DEVICE)).size() # 获取切割层输出的 feature map 的 size
logging.debug(f"{config.model_type} is trained which is splitted between layer {config.cut_point} and {config.cut_point+1}")

# 优化器，loss等
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE) # 正常 SL 训练的 loss
server_optim = torch.optim.SGD(server_globalmodel.parameters(), config.sl_lr) # 正常 SL 训练的 optim
client_optims = [torch.optim.SGD(model.parameters(), config.sl_lr) for model in client_localmodels]
logging.debug(f"split learning rate: {config.sl_lr} | discriminator learning rate: {config.adv_lr}")

# 鉴别器
if config.disc_type == "anchor":
    # 如果是锚点方式的discriminator，则目标域与每一个其他域单独组成一簇
    cluster_unions = [ClusterUnionAnchor(config.disc_mode, config.k, fm_size, config.adv_lr, DEVICE) for _ in range(num_client-1)]
    logging.debug(f"anchor discriminator worked in {config.disc_mode} mode")
elif config.disc_type == "mul":
    # 如果是多分类方式的discriminator，则只有一簇
    cluster_union = ClusterUnionMultiout(config.disc_mode, config.k, fm_size, num_client, config.adv_lr, DEVICE)
    logging.debug(f"mul-classifier discriminator worked in {config.disc_mode} mode")
elif config.disc_type == "none":
    # SL没有鉴别器
    logging.debug("vanilla split learning with no discriminator")
else:
    raise ValueError(f"Unrecognized discriminator type: `{config.disc_type}`")

if __name__ == "__main__":
    batch_counter = 0 # 控制 testing 频率的计数器

    for round in range(config.global_round):
        logging.info(f"round: {round}")

        # 下发模型
        if config.aggregate:
            logging.debug(f"distribute model")
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

############################################################ 鉴别器起作用的部分， 去掉就是单纯的SL #########################################################################
#######################################################################################################################################################################
                if config.ratio_disc < 0: # 此时如果ratio_disc是负数，会抵消对抗的效果
                    logging.warning("`ratio_disc` is negative.")

                if config.disc_type == "anchor":
                    # 各个簇的鉴别器分别更新
                    [cluster_union.update(feature_maps[config.target_domain], [feature_map]) for cluster_union, feature_map in zip(cluster_unions, feature_maps[:config.target_domain] + feature_maps[config.target_domain+1:])]
                    if config.disc_mode == "gr": # 如果鉴别器是梯度反转模式，则将各个client model上的梯度乘-1进行反转
                        [ratio_model_grad(model, -1.) if i != config.target_domain else None for i, model in enumerate(client_localmodels)]
                    # 控制鉴别器反向回来的梯度的大小
                    [ratio_model_grad(model, config.ratio_disc) if i != config.target_domain else None for i, model in enumerate(client_localmodels)]
                elif config.disc_type == "mul":
                    # 只有一个鉴别器，所以更新一次就可以了
                    cluster_union.update(feature_maps)
                    if config.disc_mode == "gr": # 如果鉴别器是梯度反转模式，则将各个client model上的梯度乘-1进行反转
                        [ratio_model_grad(model, -1.) for model in client_localmodels]
                    # 控制鉴别器反向回来的梯度的大小
                    [ratio_model_grad(model, config.ratio_disc) for model in client_localmodels]
                elif config.disc_type == "none":
                    pass
                else:
                    raise ValueError(f"Unrecognized discriminator type: `{config.disc_type}`")
#######################################################################################################################################################################

                # 将 feature map 输入 server model，并反向传播，得到 SL 的梯度，这些梯度会与之前 gan 的加权梯度加起来
                outputs = [server_globalmodel(feature_map) for feature_map in feature_maps]
                losses = [loss_fn(output, label) for output, label in zip(outputs, labels)]
                [loss.backward() for loss in losses]
                ratio_model_grad(server_globalmodel, 1/num_client) # server侧的模型上会叠加多个client的梯度，这里平均一下

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
                        logging_info += f" {datasets[i]}[{acc:.4f}] |"
                    total_acc = sum(accs)
                    logging_info += f" TOTAL_ACC[{total_acc:.4f}]"
                    logging.info(logging_info)
                    wandb.log({
                        "round": round,
                        "batch": batch_counter,
                        "total_acc": total_acc,
                        **{f"{dataset} acc": acc for dataset, acc in zip(datasets, accs)}
                    })
        # 将各个用户最后一个 batch 的第一个样本计算得到的 feature map 保存起来，用来观察 gan 对 feature map 的影响
        # 图像保存在 images 文件夹下，每个 round 保存一张
        # 图像中，每一行表示对应用户输出的各层 feature map；每一列表示 feature map 对应 channel 在不同用户上的区别
        # torchvision.utils.save_image(torch.concat([feature_map[0].reshape(fm_size[1], 1, fm_size[2], fm_size[3]) for feature_map in feature_maps], dim=0), f"images/{alg}/{round}.png", nrow=fm_size[1], normalize=True)

        # 聚合模型
        if config.aggregate:
            logging.debug("aggregate model")
            client_globalmodel = aggregate_model(client_localmodels, [1 / num_client] * num_client)
