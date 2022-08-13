# GanSFL

> 实验数据集引用 "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" 中处理后的数据集。每个客户端所持有的数据集来自不同域（分别为 MNIST | SVHN | USPS | SynthDigits | MNIST_M）。

## 运行代码

1. 安装 pytorch，torchvision，wandb
2. 因为代码中使用 wandb 进行可视化和参数管理，所以在运行前需要运行 ```wandb login``` 登陆 wandb 账号。若不想用wandb，运行 ```wandb offline``` 关掉即可。
3. 新建 datsets 文件夹，并将[下载](https://drive.google.com/file/d/1moBE_ASD5vIOaU8ZHm_Nsj0KAfX5T0Sf/view?usp=sharing)的数据集放入其中
4. 代码的配置参数都在 config-defaults.yaml 里，修改后就可以不同参数运行。

