#!/usr/bin/env bash
# =============================================================================
# FPT — FPT_PSR 神经元 Sequential CIFAR-100 训练启动脚本
#
# 默认配置对齐 FPT 论文 Table 6（Sequential CIFAR-100）：
#   epochs=256, batch=128, T=32(代码内部固定), lambda=0.5, lr=0.1, SGD, channels=128
#
# 用法：
#   ./run_fpt.sh                          # 默认 CIFAR-100，前台训练
#   CLASS_NUM=10 ./run_fpt.sh             # 训练 CIFAR-10（数据目录/日志/输出目录自动区分）
#   BG=1 ./run_fpt.sh                     # nohup 后台训练
#   NEU=FPT_adaptive_alpha ./run_fpt.sh   # 切换神经元（如 FPT_learnable_alpha/LIF/...）
#   DEVICE=cuda:2 DATA_DIR=/data/cifar100 ./run_fpt.sh
#   ./run_fpt.sh -epochs 100 -lr 0.01     # 追加/覆盖任意参数（透传给 main.py）
# =============================================================================
set -e

# 脚本所在目录，避免从其他目录调用时路径出错
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 可配置项（也可用环境变量覆盖）----------------------------------------------
DEVICE="${DEVICE:-cuda:0}"
NEU="${NEU:-FPT_PSR}"
EPOCHS="${EPOCHS:-256}"
BATCH="${BATCH:-128}"
LR="${LR:-0.1}"
CHANNELS="${CHANNELS:-128}"
OPT="${OPT:-sgd}"
CLASS_NUM="${CLASS_NUM:-100}"
J="${J:-4}"
AMP="${AMP:-true}"
BG="${BG:-0}"

# 数据目录：未显式设置时，按 CLASS_NUM 自动选本仓库相对路径
#   CIFAR-10  -> ../datasets/CIFAR10   （d:\Code\lw260109\code\Parallel-Spiking-Neuron\datasets\CIFAR10）
#   CIFAR-100 -> ../datasets/CIFAR100
# 服务器/自定义路径请用 DATA_DIR 覆盖，如 DATA_DIR=/data/Datasets/cifar-100-python
if [[ -z "${DATA_DIR:-}" ]]; then
    DATA_DIR="../datasets/CIFAR${CLASS_NUM}"
fi
# 输出根目录：也按 CLASS_NUM 区分，避免 c10/c100 实验混在一起
if [[ -z "${OUT_DIR:-}" ]]; then
    OUT_DIR="./logs_c${CLASS_NUM}"
fi
# --------------------------------------------------------------------------------

mkdir -p logs
LOG="logs/${NEU}_c${CLASS_NUM}_e${EPOCHS}_b${BATCH}_${OPT}_lr${LR}_c${CHANNELS}.log"

CMD=(python -u main.py
     -device "$DEVICE" -data-dir "$DATA_DIR" -out-dir "$OUT_DIR"
     -neu "$NEU" -epochs "$EPOCHS" -b "$BATCH" -lr "$LR" -channels "$CHANNELS"
     -opt "$OPT" -class-num "$CLASS_NUM" -j "$J" -amp "$AMP" "$@")

echo ">>> FPT run: ${CMD[*]}"
echo ">>> log: $LOG"

if [[ "$BG" == "1" ]]; then
    # 后台运行，日志实时落盘，可用 tail -f 查看
    nohup "${CMD[@]}" > "$LOG" 2>&1 &
    echo ">>> started in background, pid=$!"
    echo ">>> watch: tail -f $LOG"
else
    "${CMD[@]}" > "$LOG" 2>&1
    echo ">>> finished, see $LOG"
fi