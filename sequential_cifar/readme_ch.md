## 使用方法

本仓库中的 `glif.py` 是基于 GLIF 论文 [GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks](https://openreview.net/forum?id=UmFSx2c4ubT) 修改的。

请注意，论文中神经元的名称与代码中的名称不完全一致。以下是名称对照表：

| 论文中的神经元      | 代码中的神经元 |
| ------------ | ------- |
| PSN          | if5     |
| masked PSN   | if5pmd8 |
| sliding PSN  | mspsn   |
| LIF wo reset | lifnr   |

### 命令行参数

```
用法: train_secf10.py [-h] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                       [-resume RESUME] [-amp] [-opt OPT] [-momentum MOMENTUM] [-lr LR] [-channels CHANNELS]
                       [-neu NEU] [-class-num CLASS_NUM] [-P P] [-exp-init]

顺序 CIFAR10/100 分类

参数:
  -h, --help            显示帮助信息并退出
  -device DEVICE        设备
  -b B                  批次大小
  -epochs N             训练的总轮数
  -j N                  数据加载的工作进程数（默认: 4）
  -data-dir DATA_DIR    CIFAR10/100 数据集的根目录
  -out-dir OUT_DIR      保存日志和检查点的根目录
  -resume RESUME        从检查点路径恢复训练
  -amp                  自动混合精度训练
  -opt OPT              使用哪种优化器，SDG 或 Adam
  -momentum MOMENTUM    SGD 的动量
  -lr LR                学习率
  -channels CHANNELS    CSNN 的通道数
  -neu NEU              使用哪种神经元
  -class-num CLASS_NUM 类别数量
  -P P                  masked/sliding PSN 的阶数
  -exp-init             使用指数初始化方法初始化 SPSN 的权重
```

### 使用示例

在顺序 CIFAR10 上使用 PSN：

```bash
# cifar10
python train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5

python train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5 

# cifar100
python train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5 

python train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5
```

在顺序 CIFAR100 上使用 32 阶 SPSN：

```bash
python train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu mspsn -P 32

python train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu mspsn -P 32 

python train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32

python train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.002 -channels 128 -epochs 256 -neu mspsn -P 32
```

在顺序 CIFAR100 上使用 32 阶 masked PSN：

```bashs
python train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32

python train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32 

python train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32

python train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp  -channels 128 -epochs 256 -neu if5pmd8 -P 32 -opt adamw -lr 0.002 
```

