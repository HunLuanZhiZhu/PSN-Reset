@REM CRR: Conv-BN-ReLU IFNode5, support -r to control whether to use learnable reset

@REM no reset version
mkdir logscrr10 2>nul
python -u train_secf10crr.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscrr10\if5.log 2>&1

mkdir logscrr100 2>nul
python -u train_secf10crr.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscrr100\if5.log 2>&1

@REM reset version with -r
mkdir logscrr10 2>nul
python -u train_secf10crr.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 -r > logscrr10\if5-r.log 2>&1

mkdir logscrr100 2>nul
python -u train_secf10crr.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 -r > logscrr100\if5-r.log 2>&1
