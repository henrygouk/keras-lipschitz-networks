#!/bin/bash

#for i in $(seq 1 5)
#do
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/none.txt
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/sd.txt --sd-conv=0.01 --sd-dense=0.01
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/batchnorm.txt --batchnorm
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/dropout.txt --drop-conv=0.41 --drop-dense=0.1
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/batchnorm-dropout.txt --batchnorm --drop-conv=0.2 --drop-dense=0.5

    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l1.txt --lcc=1 --lambda-conv=56 --lambda-dense=48
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l2.txt --lcc=2 --lambda-conv=4.3 --lambda-dense=7.6
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/linf.txt --lcc=inf --lambda-conv=54 --lambda-dense=30

    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l1-batchnorm.txt --lcc=1 --lambda-conv=23 --lambda-dense=52 --lambda-bn=8.4 --batchnorm
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l2-batchnorm.txt --lcc=2 --lambda-conv=1.9 --lambda-dense=6.1 --lambda-bn=5.7 --batchnorm
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/linf-batchnorm.txt --lcc=inf --lambda-conv=20 --lambda-dense=56 --lambda-bn=4.7 --batchnorm

    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l1-dropout.txt --lcc=1 --lambda-conv=62 --lambda-dense=65 --drop-conv=0.15 --drop-dense=0.22
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l2-dropout.txt --lcc=2 --lambda-conv=5.2 --lambda-dense=10 --drop-conv=0.43 --drop-dense=0.32
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/linf-dropout.txt --lcc=inf --lambda-conv=56 --lambda-dense=51 --drop-conv=0.27 --drop-dense=0.21

    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l1-batchnorm-dropout.txt --lcc=1 --lambda-conv=20 --lambda-dense=36 --lambda-bn=12 --batchnorm --drop-conv=0.35 --drop-dense=0.3
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l2-batchnorm-dropout.txt --lcc=2 --lambda-conv=1.2 --lambda-dense=10 --lambda-bn=11 --batchnorm --drop-conv=0.14 --drop-dense=0.38
    ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/linf-batchnorm-dropout.txt --lcc=inf --lambda-conv=20 --lambda-dense=24 --lambda-bn=7.2 --batchnorm --drop-conv=0.46 --drop-dense=0.49
#done
