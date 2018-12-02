#!/bin/bash

./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/none.txt
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/sd.txt --sd-conv=0.01 --sd-dense=0.01
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/batchnorm.txt --batchnorm
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/dropout.txt --drop-conv=0.2 --drop-dense=0.5
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/batchnorm-dropout.txt --batchnorm --drop-conv=0.2 --drop-dense=0.5

./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l1.txt --lcc=1 --lambda-conv=48 --lambda-dense=48
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l2.txt --lcc=2 --lambda-conv=8 --lambda-dense=8
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/linf.txt --lcc=inf --lambda-conv=48 --lambda-dense=16

./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l1-batchnorm.txt --lcc=1 --lambda-conv=20 --lambda-dense=20 --lambda-bn=20 --batchnorm
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l2-batchnorm.txt --lcc=2 --lambda-conv=1.5 --lambda-dense=1.5 --lambda-bn=8 --batchnorm
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/linf-batchnorm.txt --lcc=inf --lambda-conv=20 --lambda-dense=4 --lambda-bn=20 --batchnorm

./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l1-dropout.txt --lcc=1 --lambda-conv=60 --lambda-dense=60 --drop-conv=0.2 --drop-dense=0.5
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l2-dropout.txt --lcc=2 --lambda-conv=10 --lambda-dense=10 --drop-conv=0.2 --drop-dense=0.5
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/linf-dropout.txt --lcc=inf --lambda-conv=60 --lambda-dense=20 --drop-conv=0.2 --drop-dense=0.5

./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l1-batchnorm-dropout.txt --lcc=1 --lambda-conv=28 --lambda-dense=28 --lambda-bn=28 --batchnorm --drop-conv=0.2 --drop-dense=0.5
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/l2-batchnorm-dropout.txt --lcc=2 --lambda-conv=2 --lambda-dense=2 --lambda-bn=8 --batchnorm --drop-conv=0.2 --drop-dense=0.5
./cifar10.py --arch=vgg --log-path=results/cifar10/vgg/linf-batchnorm-dropout.txt --lcc=inf --lambda-conv=10 --lambda-dense=6 --lambda-bn=10 --batchnorm --drop-conv=0.2 --drop-dense=0.5
