#!/bin/bash

for i in 0.01 0.02 #0.1 0.2 0.3 0.4 0.5
do
    ./cifar10.py --arch=vgg --log-path=results/cifar10/vgg-sample/none.txt --subsample=$i
    ./cifar10.py --arch=vgg --log-path=results/cifar10/vgg-sample/sd.txt --sd-conv=0.01 --sd-dense=0.01 --subsample=$i
    ./cifar10.py --arch=vgg --log-path=results/cifar10/vgg-sample/batchnorm.txt --batchnorm --subsample=$i
    ./cifar10.py --arch=vgg --log-path=results/cifar10/vgg-sample/dropout.txt --drop-conv=0.2 --drop-dense=0.5 --subsample=$i
    ./cifar10.py --arch=vgg --log-path=results/cifar10/vgg-sample/l1-batchnorm.txt --lcc=1 --lambda-conv=20 --lambda-dense=20 --lambda-bn=20 --batchnorm --subsample=$i
    ./cifar10.py --arch=vgg --log-path=results/cifar10/vgg-sample/l2-batchnorm.txt --lcc=2 --lambda-conv=1.5 --lambda-dense=1.5 --lambda-bn=8 --batchnorm --subsample=$i
    ./cifar10.py --arch=vgg --log-path=results/cifar10/vgg-sample/linf-batchnorm.txt --lcc=inf --lambda-conv=20 --lambda-dense=4 --lambda-bn=20 --batchnorm --subsample=$i
done
