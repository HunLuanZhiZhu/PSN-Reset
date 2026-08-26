mkdir logs10 2>nul
mkdir logs100 2>nul


@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5 > logs10\if5.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5 > logs100\if5.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32  > logs10\mspsn_adamw.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32 > logs100\mspsn_adamw.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32  > logs10\if5pmd8.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32 > logs100\if5pmd8.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu lifnr > logs10\lifnr.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu lifnr > logs100\lifnr.log 2>&1



@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu lif > logs10\lif.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu lif > logs100\lif.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu plif > logs10\plif.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu plif > logs100\plif.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu klif > logs10\klif.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu klif > logs100\klif.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu glif > logs10\glif.log 2>&1

@REM python -u train_secf10.py -data-dir ../datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu glif > logs100\glif.log 2>&1




