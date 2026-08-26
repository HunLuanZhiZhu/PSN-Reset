

@REM mkdir logscq10 2>nul

@REM python -u train_secf10cq.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logscq10\if5.log 2>&1


@REM mkdir logscq100 2>nul

@REM python -u train_secf10cq.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logscq100\if5.log 2>&1


@REM mkdir logscqr100 2>nul

@REM python -u train_secf10cqr.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logscqr100\if5.log 2>&1

@REM mkdir logscqr10 2>nul

@REM python -u train_secf10cqr.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logscqr10\if5.log 2>&1


mkdir logscq10 2>nul

python -u train_secf10cq.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 1e-3 -channels 256 -epochs 5 -neu if5 > logscqr10\test.log 2>&1

