mkdir logsr10 2>nul
mkdir logsr100 2>nul

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5 > logsr10\if5.log 2>&1

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5 > logsr100\if5.log 2>&1

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu mspsn -P 32  > logsr10\mspsn_sgd.log 2>&1
python -u train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32  > logsr10\mspsn_adamw.log 2>&1

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu mspsn -P 32 > logsr100\mspsn_sgd.log 2>&1
python -u train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32 > logsr100\mspsn_adamw.log 2>&1

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32  > logsr10\if5pmd8.log 2>&1

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32 > logsr100\if5pmd8.log 2>&1

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu lifnr > logsr10\lifnr.log 2>&1

@REM python -u train_secf10r.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu lifnr > logsr100\lifnr.log 2>&1

