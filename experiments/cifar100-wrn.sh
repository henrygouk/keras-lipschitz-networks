#!/bin/bash

#for i in $(seq 1 5)
#do
    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/none.txt
    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/sd.txt --sd-conv=0.01 --sd-dense=0.1
    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/dropout.txt --drop-conv=0.17

    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/l1.txt --lcc=1 --lambda-conv=87 --lambda-dense=47 --lambda-bn=9.4
    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/l2.txt --lcc=2 --lambda-conv=10 --lambda-dense=3.9 --lambda-bn=6.1
    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/linf.txt --lcc=inf --lambda-conv=87 --lambda-dense=38 --lambda-bn=11

    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/l1-dropout.txt --lcc=1 --lambda-conv=80 --lambda-dense=21 --lambda-bn=10 --drop-conv=0.1 --drop-dense=0.23
    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/l2-dropout.txt --lcc=2 --lambda-conv=12 --lambda-dense=3.8 --lambda-bn=3.5 --drop-conv=0.33 --drop-dense=0.45
    ./cifar100.py --arch=wrn --log-path=results/cifar100/wrn/linf-dropout.txt --lcc=inf --lambda-conv=77 --lambda-dense=59 --lambda-bn=5.5 --drop-conv=0.32 --drop-dense=0.33
#done