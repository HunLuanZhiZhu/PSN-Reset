# PSN-Reset

> Research code associated with a **submitted manuscript** on parallel reset mechanisms for spiking neural networks.  
> This repository contains implementations and experiment scripts for restoring historical-output feedback in parallel spiking neurons while preserving parallel computation across the temporal dimension.

## Overview

Parallel spiking neurons (PSNs), including masked PSN and sliding PSN variants, improve temporal parallelism by removing the recurrent reset dependency that appears in conventional spiking neurons.

That design has an important consequence: once reset is removed, the effect of previous outputs on later membrane states is also removed.

This repository studies how to **reintroduce reset-like historical feedback without returning to an explicitly timestep-by-timestep recurrent implementation**.

Two reset mechanisms are implemented and evaluated:

- **PSR — Prefix-Scan Soft Reset**  
  Re-expresses soft-reset feedback using exponentially weighted prefix-style cumulative operations. Instead of updating the membrane state one timestep at a time, the historical reset contribution is constructed with vectorized cumulative computation over the time dimension.

- **CCR — Channel-Wise Convolutional Reset**  
  Learns historical-output feedback with causal depthwise temporal convolution. Each channel owns its reset kernel independently through grouped `Conv1d` (`groups = C`), so reset feedback is modeled without cross-channel mixing.

The experiments also study several output representations, including:

- binary spikes;
- power-of-two multi-level outputs;
- complex power-of-two outputs;
- continuous / ReLU-style outputs.

The current code covers **Sequential CIFAR-10**, **Sequential CIFAR-100**, and standard **CIFAR-10** classification experiments.

---

## Research motivation

A conventional soft-reset neuron contains an explicit temporal dependency:

```text
previous state + current input
            │
            ▼
      membrane state
            │
            ▼
          output
            │
            └────────► reset contribution to future states
```

Removing reset breaks this feedback path and makes the temporal computation easier to parallelize:

```text
input sequence
     │
     ▼
parallel temporal transform
     │
     ▼
all membrane states / outputs
```

PSN-Reset asks whether we can keep the second computation pattern while recovering part of the first neuron's historical feedback.

Conceptually:

```text
                    historical-output feedback
                              │
                              ▼
input sequence ──► parallel temporal transform ──► output sequence
                       ▲
                       │
                PSR or learned CCR
```

The repository is therefore organized as an experimental codebase around a specific research question rather than as a general-purpose SNN framework.

---

## Methods

### PSR: Prefix-Scan Soft Reset

The PSR implementation uses exponential weighting together with cumulative sums over the temporal dimension.

A simplified view of the implementation is:

```python
feedback = spike_history * decay_weights
feedback = feedback.cumsum(dim=0)
feedback = feedback * inverse_decay_weights
membrane[1:] -= feedback
```

The actual implementation performs an additional correction pass so that the reset feedback is computed from updated membrane states rather than from a single uncorrected estimate.

The key point is that reset is expressed through **tensor-wide cumulative operations** instead of an explicit Python loop over timesteps.

Relevant implementations:

- `sequential_cifar/train_secf10r.py`
- `cifar10/train_cf10r.py`
- related sequential-CIFAR variants containing the reset operator

### CCR: Channel-Wise Convolutional Reset

CCR treats reset feedback as a learnable causal temporal filter.

The central module is a depthwise temporal convolution:

```python
nn.Conv1d(
    in_channels=C,
    out_channels=C,
    kernel_size=K,
    groups=C,
    bias=False,
)
```

Using `groups=C` gives every channel its own temporal reset kernel while avoiding cross-channel mixing.

The reset contribution is computed from past outputs and subtracted from the unreset membrane sequence:

```text
unreset membrane
      │
      ├───────────────┐
      │               ▼
      │        historical outputs
      │               │
      │               ▼
      │        depthwise causal conv
      │               │
      │               ▼
      └─────── minus reset feedback
                      │
                      ▼
               corrected membrane
```

Relevant implementation:

- `sequential_cifar/train_secf10cr.py`

---

## Experimental scope

The repository currently contains two main experiment families.

### Sequential CIFAR-10 / CIFAR-100

Directory:

```text
sequential_cifar/
```

This part contains the main sequence-classification experiments and several neuron / output / reset variants.

The naming used in code is partly inherited from earlier experiment scripts. The main neuron-name mapping is:

| Method name | Code name |
|---|---|
| PSN | `if5` |
| masked PSN | `if5pmd8` |
| sliding PSN | `mspsn` |
| LIF without reset | `lifnr` |

See `sequential_cifar/readme.md` for the command-line interface and experiment examples.

### CIFAR-10

Directory:

```text
cifar10/
```

This part provides standard CIFAR-10 experiments for comparing PSN behavior with and without the proposed reset mechanism under a shorter temporal horizon.

See `cifar10/readme.md` for detailed arguments.

---

## Output representations

In addition to reset mechanisms, the experiment scripts explore different forms of neuron output.

| Representation | Purpose |
|---|---|
| Binary spike | Standard discrete spiking output |
| Power-of-two multi-level output | Quantized multi-level response with power-of-two structure |
| Complex power-of-two output | Complex-valued temporal representation with quantized output structure |
| Continuous / ReLU-style output | Non-binary reference for studying the effect of output discretization |

Some experiment files contain true complex-valued operations, including complex temporal convolution and complex linear layers.

These variants are kept in the repository because the submitted work studies the interaction between **reset feedback** and **output representation**, rather than treating reset as an isolated implementation detail.

---

## Repository structure

```text
PSN-Reset/
├── README.md
├── README_ch.md
├── LICENSE
├── requirements.txt
│
├── sequential_cifar/
│   ├── train_secf10.py
│   ├── train_secf10r.py
│   ├── train_secf10c.py
│   ├── train_secf10cr.py
│   ├── train_secf10cq.py
│   ├── train_secf10cqr.py
│   ├── train_secf10cqrc.py
│   ├── train_secf10crk.py
│   ├── train_secf10crr.py
│   ├── glif.py
│   ├── readme.md
│   ├── readme_ch.md
│   ├── FPT/
│   └── *.cmd
│
└── cifar10/
    ├── train_cf10.py
    ├── train_cf10r.py
    ├── readme.md
    └── readme_zh.md
```

The repository preserves several experiment-specific scripts rather than forcing them into one abstraction layer. This makes the exact configuration used by a particular experiment easier to inspect, at the cost of some duplicated training code.

---

## Installation

Recommended dependencies are listed in `requirements.txt`:

```text
torch>=2.0
torchvision>=0.15
spikingjelly==0.0.0.0.14
tensorboard
```

Install with:

```bash
pip install -r requirements.txt
```

A CUDA-capable environment is recommended for reproducing the training experiments.

---

## Reproducing experiments

### Sequential CIFAR-10 with PSN

```bash
cd sequential_cifar

python train_secf10.py \
  -data-dir ../datasets/CIFAR10 \
  -amp \
  -opt sgd \
  -channels 128 \
  -epochs 256 \
  -neu if5
```

### Sequential CIFAR-100 with sliding PSN

```bash
python train_secf10.py \
  -data-dir ../datasets/CIFAR100 \
  -class-num 100 \
  -amp \
  -opt adamw \
  -lr 0.001 \
  -channels 128 \
  -epochs 256 \
  -neu mspsn \
  -P 32
```

### CIFAR-10

```bash
cd ../cifar10

python train_cf10.py \
  -data-dir ../datasets/CIFAR10 \
  -amp \
  -opt sgd \
  -channels 256 \
  -epochs 1024 \
  -T 4
```

For reset-enabled and representation-specific configurations, use the corresponding experiment script and refer to the README inside each experiment directory.

---

## Reproducibility notes

This repository contains the research code associated with a **submitted manuscript**.

A few details are worth noting when reproducing experiments:

- training scripts are experiment-oriented and may contain configuration differences between variants;
- filenames preserve the naming used during the research process;
- dataset paths and output directories are supplied through command-line arguments;
- checkpoints, datasets, TensorBoard logs, and other large generated artifacts are intentionally excluded from Git;
- the manuscript contains the broader experimental context and quantitative comparison associated with these implementations.

If you are reproducing a specific configuration, use the exact script corresponding to that configuration rather than assuming all variants share an identical implementation path.

---

## External code and dependencies

This project builds on the PyTorch / SpikingJelly ecosystem.

The file:

```text
sequential_cifar/glif.py
```

is modified from the implementation associated with:

**GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks**  
OpenReview: https://openreview.net/forum?id=UmFSx2c4ubT

Relevant external project:

- SpikingJelly: https://github.com/fangwei123456/spikingjelly

The PSN reset mechanisms and the experiment variants in this repository are organized around the research described above; third-party components retain their original attribution.

---

## Paper status and citation

This repository currently accompanies a **submitted manuscript**.

The public citation entry will be added after bibliographic information suitable for citation becomes available.

Until then, if you use this code in research, please reference this repository and the upstream works on which the experiments depend.

---

## License

This repository is released under the **MIT License**. See [LICENSE](./LICENSE).

