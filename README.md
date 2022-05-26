# GanSFL

> 实验数据集引用 "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" 中处理后的数据集。每个客户端所持有的数据集来自不同域（分别为 MNIST | SVHN | USPS | SynthDigits | MNIST_M）。

## 运行代码

1. 安装 pytorch，torchvision，wandb
2. 因为代码中使用 wandb 进行可视化和参数管理，所以在运行前需要运行 ```wandb login``` 登陆 wandb 账号。若不想用wandb，运行 ```wandb offline``` 关掉即可。
3. 新建 data 文件夹，并将[下载](https://drive.google.com/file/d/1moBE_ASD5vIOaU8ZHm_Nsj0KAfX5T0Sf/view?usp=sharing)的数据集放入其中
4. 在代码目录里新建 images/GanSFL 和 images/SFL 文件夹，用于保存训练过程中输出的 feature map
5. 代码的配置参数都在 config-defaults.yaml 里，修改后就可以不同参数运行。

## GanSFL 与 SFL 的对比

> 目前添加了鉴别器后，模型收敛的性能反而变差了。[详细对比结果](https://wandb.ai/sjinglong/GanSFL/sweeps/8xi7zmmj?workspace=user-sjinglong)

![](./assets/alg.png)

## 在不同点切分模型对 GanSFL 的影响

> 采用的模型是 resnet18, 所以切割是按 block 切割的。而模型输入处的卷积层和输出处的线性层也作为一个 block，所以总共有10个block。实验中分别将切割点设为：1，2，3，4，5，6，7，8，9。[详细对比结果](https://wandb.ai/sjinglong/GanSFL/sweeps/bt207pyr?workspace=user-sjinglong)

<!-- ![](./assets/cutpoint1.png)
![](./assets/cutpoint2.png)
![](./assets/cutpoint3.png)
![](./assets/cutpoint4.png)
![](./assets/cutpoint5.png) -->

## 客户端数据集大小对 GanSFL 的影响

> 分别使用原数据集的：10%，30%，50%，70%，100%进行训练。[详细对比结果](https://wandb.ai/sjinglong/GanSFL/sweeps/1lod8gja?workspace=user-sjinglong)


## 判别器反向传播在客户端模型上的梯度的权重对 GanSFL 的影响

> SFL的梯度为g1，鉴别器的梯度为g2，则客户端在更新参数时的梯度为g1 + ratio_gan*g2。实验中分别设置 ratio_gan 为：0.01，0.05，0.1，0.2。[详细对比结果](https://wandb.ai/sjinglong/GanSFL/sweeps/54k1qtl0?workspace=user-sjinglong)

## 以不同 client 输出的 feature map 作为目标域对 GanSFL 的影响

> 实验中分别以client 0， client 1，client 2，client 3，client 4的 feature map 作为目标域进行训练。[详细对比结果](https://wandb.ai/sjinglong/GanSFL/sweeps/61s6lk59?workspace=user-sjinglong)