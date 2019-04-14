#!/bin/bash

for i in 0.1 0.2 0.3 0.4 0.5
do
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn-sample/none.txt --subsample=$i
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn-sample/sd.txt --sd-conv=0.01 --sd-dense=0.01 --subsample=$i
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn-sample/dropout.txt --drop-conv=0.2 --subsample=$i
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn-sample/l1.txt --lcc=1 --lambda-conv=64 --lambda-dense=64 --lambda-bn=10 --subsample=$i
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn-sample/l2.txt --lcc=2 --lambda-conv=7 --lambda-dense=7 --lambda-bn=10 --subsample=$i
    ./cifar10.py --arch=wrn --log-path=results/cifar10/wrn-sample/linf.txt --lcc=inf --lambda-conv=52 --lambda-dense=38 --lambda-bn=10 --subsample=$i
done