## 使用方法

```bash
用法: train_cf10.py [-h] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                     [-resume RESUME] [-amp] [-opt OPT] [-momentum MOMENTUM] [-lr LR] [-channels CHANNELS] [-T T]

CIFAR10 分类

选项:
  -h, --help          显示帮助信息并退出
  -device DEVICE      设备
  -b B                批次大小
  -epochs N           训练的总轮数
  -j N                数据加载工作进程数（默认为 4）
  -data-dir DATA_DIR  CIFAR10 数据集的根目录
  -out-dir OUT_DIR    保存日志和检查点的根目录
  -resume RESUME      从检查点路径恢复训练
  -amp                自动混合精度训练
  -opt OPT            使用哪个优化器。SGD 或 AdamW
  -momentum MOMENTUM  SGD 的动量
  -lr LR              学习率
  -channels CHANNELS  CSNN 的通道数
  -T T                时间步数
```

论文中使用的选项：

```
python train_cf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -device cuda:0 -T 4

python -m train_cf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -device cuda:0 -T 4 | Tee-Object -FilePath logr.txt

python -m train_cf10c.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -device cuda:0 -T 4 | Tee-Object -FilePath logc.txt

python -m train_cf10cr.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -device cuda:0 -T 4 | Tee-Object -FilePath logcr.txt
```

原始训练参数：

```
Namespace(device='cuda:1', b=128, epochs=1024, j=4, data_dir='/datasets/CIFAR10', out_dir='./logs_cf10cr', resume=None, amp=True, opt='sgd', momentum=0.9, lr=0.1, channels=256, T=4, neu='if5', sg='atan')
train_cf10.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -neu if5 -device cuda:1 -T 4
```

注意：PSN 在代码中被命名为 IF5。
