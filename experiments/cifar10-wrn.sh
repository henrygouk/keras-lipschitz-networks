#!/bin/bash

for i in $(seq 1 5)
do
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/none.txt
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/sd.txt --sd-conv=0.01 --sd-dense=0.01
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/dropout.txt --drop-conv=0.2

    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l1.txt --lcc=1 --lambda-conv=64 --lambda-dense=64 --lambda-bn=10
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l2.txt --lcc=2 --lambda-conv=7 --lambda-dense=7 --lambda-bn=10
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/linf.txt --lcc=inf --lambda-conv=52 --lambda-dense=38 --lambda-bn=10

    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l1-dropout.txt --lcc=1 --lambda-conv=72 --lambda-dense=64 --lambda-bn=10 --drop-conv=0.2
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l2-dropout.txt --lcc=2 --lambda-conv=8 --lambda-dense=7 --lambda-bn=10 --drop-conv=0.2
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/linf-dropout.txt --lcc=inf --lambda-conv=60 --lambda-dense=38 --lambda-bn=10 --drop-conv=0.2
done