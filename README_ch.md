# PSN-Reset

面向并行脉冲神经元的并行重置机制，基于 PyTorch 与 SpikingJelly 实现。

并行脉冲神经元（PSN、masked PSN、sliding PSN 等）通过移除 reset 操作，使全部时间步的膜电位可以并行计算。然而，移除 reset 也同时舍弃了历史输出对后续状态的反馈。本仓库在不破坏时间维并行性的前提下，通过以下两种机制重新引入历史输出反馈：

- **PSR（Prefix-Scan Soft Reset，前缀扫描软重置）**：将标准 LIF soft reset 实现为带权重的*排他前缀扫描*。通过前缀和形式的并行计算而非顺序递推，在有限轮并行迭代后恢复 soft reset 的指数衰减反馈。
- **CCR（Channel-Wise Convolutional Reset，逐通道卷积重置）**：将历史输出反馈扩展到卷积型并行神经元（如 sliding PSN），通过可学习的逐通道因果卷积实现反馈，不引入通道间的信息混合。

本仓库还比较了不同的输出表示——二值脉冲、Power-of-two 多级输出、复数 Power-of-two 输出以及连续 ReLU 激活——在 Sequential CIFAR-10/100 与常规 CIFAR-10 分类任务上的表现。

## 目录结构

```
PSN-Reset/
├── sequential_cifar/   # Sequential CIFAR-10/100 训练代码
│   ├── train_secf10*.py    # 各配置的训练脚本
│   ├── glif.py             # 修改自 GLIF 项目（见 readme.md）
│   └── *.cmd               # Windows 批处理脚本
├── cifar10/            # 常规 CIFAR-10 训练代码
│   ├── train_cf10.py       # 训练脚本
│   ├── train_cf10r.py      # 训练脚本（reset 版本）
│   └── readme.md / readme_zh.md
└── README.md
```

## 环境

```bash
pip install -r requirements.txt
```

## 使用

进入对应实验目录，参见其中的 `readme.md`（中文见 `readme_zh.md` / `readme_ch.md`），包含完整的命令行参数说明与使用实例。

```bash
# Sequential CIFAR-10
cd sequential_cifar
python train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5

# Sequential CIFAR-100
python train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32

# 常规 CIFAR-10
cd cifar10
python train_cf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -T 4
```

## 说明

- 训练脚本中的神经元名称与方法名称不完全一致（例如 `if5` 对应 PSN），对应关系见 `sequential_cifar/readme.md` 中的名称对照表。
- `glif.py` 修改自 [GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks](https://openreview.net/forum?id=UmFSx2c4ubT)。
- 本仓库依赖 [SpikingJelly](https://github.com/fangwei123456/spikingjelly)。

## License

本项目采用 [MIT 许可证](LICENSE)。