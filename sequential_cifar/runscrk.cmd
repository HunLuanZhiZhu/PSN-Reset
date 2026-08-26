@REM CRK: ResetModule with fixed self.K=3

mkdir logscrk10 2>nul
python -u train_secf10crk.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscrk10\if5.log 2>&1

mkdir logscrk100 2>nul
python -u train_secf10crk.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu if5 > logscrk100\if5.log 2>&1

@REM mkdir logscrk10 2>nul
@REM python -u train_secf10crk.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32 > logscrk10\mspsn.log 2>&1

@REM mkdir logscrk100 2>nul
@REM python -u train_secf10crk.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32 > logscrk100\mspsn.log 2>&1
