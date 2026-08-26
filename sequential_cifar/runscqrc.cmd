@REM Complex IFNode5: 2C channels [re, im], spike = floor_to_pow2(h) element-wise
@REM FR = (re>0 + im>0) / complex_elements, can exceed 100%

@REM mkdir logscqrc10 2>nul
@REM python -u train_secf10cqrc.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscqrc10\if5.log 2>&1

@REM mkdir logscqrc100 2>nul
@REM python -u train_secf10cqrc.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscqrc100\if5.log 2>&1

mkdir logscqrc10 2>nul
python -u train_secf10cqrc.py -r -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscqrc10\if5-r.log 2>&1

mkdir logscqrc100 2>nul
python -u train_secf10cqrc.py -r -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscqrc100\if5-r.log 2>&1
