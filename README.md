# PSN-Reset

Parallel reset mechanisms for parallel spiking neurons, implemented with PyTorch and SpikingJelly.

Parallel spiking neurons (PSN, masked PSN, sliding PSN, etc.) remove the reset operation so that the membrane potentials of all time steps can be computed in parallel. However, removing reset also discards the feedback of historical outputs on subsequent states. This project reintroduces such historical-output feedback while keeping the time dimension fully parallel, through two mechanisms:

- **PSR (Prefix-Scan Soft Reset)**: implements the standard soft reset of LIF neurons as a weighted *exclusive prefix scan*. The exponential feedback of soft reset is recovered through prefix-sum style parallel computation instead of sequential recurrence, after a bounded number of parallel iterations.
- **CCR (Channel-Wise Convolutional Reset)**: extends historical-output feedback to convolution-based parallel neurons (e.g., sliding PSN) through learnable channel-wise causal convolutions, without mixing information across channels.

The reset mechanisms are studied together with different output representations — binary spikes, power-of-two multi-level outputs, complex power-of-two outputs, and continuous ReLU activation — on Sequential CIFAR-10/100 and CIFAR-10 classification tasks.

## Directory Structure

```
PSN-Reset/
├── sequential_cifar/   # Sequential CIFAR-10/100 training code
│   ├── train_secf10*.py    # training scripts for different configurations
│   ├── glif.py             # modified from the GLIF project (see readme.md)
│   └── *.cmd               # Windows batch scripts
├── cifar10/            # CIFAR-10 training code
│   ├── train_cf10.py       # training script
│   ├── train_cf10r.py      # training script (reset)
│   └── readme.md / readme_zh.md
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Enter the corresponding experiment directory and refer to its `readme.md` (Chinese: `readme_zh.md` / `readme_ch.md`) for the full command-line arguments and examples.

```bash
# Sequential CIFAR-10
cd sequential_cifar
python train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5

# Sequential CIFAR-100
python train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32

# CIFAR-10
cd cifar10
python train_cf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -T 4
```

## Notes

- The neuron names used in the training scripts differ from the method names above (e.g., `if5` corresponds to PSN). The mapping is listed in `sequential_cifar/readme.md`.
- `glif.py` is modified from [GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks](https://openreview.net/forum?id=UmFSx2c4ubT).
- This project depends on [SpikingJelly](https://github.com/fangwei123456/spikingjelly).

## License

This project is licensed under the [MIT License](LICENSE).