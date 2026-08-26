@REM mkdir logsc10 2>nul

@REM python -u train_secf10c.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logsc10\if5.log 2>&1

@REM mkdir logsc100 2>nul

@REM python -u train_secf10c.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logsc100\if5.log 2>&1


@REM mkdir logscr100 2>nul

@REM python -u train_secf10cr.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logscr100\if5.log 2>&1

@REM mkdir logscr10 2>nul

@REM python -u train_secf10cr.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 1e-3 -channels 128 -epochs 256 -neu if5 > logscr10\if5.log 2>&1

mkdir logscr10 2>nul
python -u train_secf10cr.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32  > logscr10\mspsn.log 2>&1

mkdir logscr100 2>nul
python -u train_secf10cr.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32 > logscr100\mspsn.log 2>&1
