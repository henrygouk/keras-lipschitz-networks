#!/bin/bash

./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/none.txt
./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/sd.txt --sd-conv=0.01 --sd-dense=0.01
./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/dropout.txt --drop-conv=0.2

./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l1.txt --lcc=1 --lambda-conv=56 --lambda-dense=56 --lambda-bn=10
./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l2.txt --lcc=2 --lambda-conv=8 --lambda-dense=8 --lambda-bn=10
./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/linf.txt --lcc=inf --lambda-conv=48 --lambda-dense=16 --lambda-bn=10

./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l1-dropout.txt --lcc=1 --lambda-conv=60 --lambda-dense=60 --drop-conv=0.2
./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/l2-dropout.txt --lcc=2 --lambda-conv=10 --lambda-dense=10 --drop-conv=0.2
./cifar10.py --arch=wrn --log-path=results/cifar10/wrn/linf-dropout.txt --lcc=inf --lambda-conv=60 --lambda-dense=20 --drop-conv=0.2
