#!/bin/bash

#for i in $(seq 1 5)
#do
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/none.txt
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/sd.txt --sd-conv=0.001 --sd-dense=0.01
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/batchnorm.txt --batchnorm
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/dropout.txt --drop-conv=0.23 --drop-dense=0.49
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/batchnorm-dropout.txt --batchnorm --drop-conv=0.47 --drop-dense=0.24

    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l1.txt --lcc=1 --lambda-conv=28 --lambda-dense=27
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l2.txt --lcc=2 --lambda-conv=4.1 --lambda-dense=1.1
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/linf.txt --lcc=inf --lambda-conv=20 --lambda-dense=50

    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l1-batchnorm.txt --lcc=1 --lambda-conv=28 --lambda-dense=40 --lambda-bn=5.6 --batchnorm
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l2-batchnorm.txt --lcc=2 --lambda-conv=4.2 --lambda-dense=3.1 --lambda-bn=4.3 --batchnorm
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/linf-batchnorm.txt --lcc=inf --lambda-conv=23 --lambda-dense=25 --lambda-bn=2.1 --batchnorm

    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l1-dropout.txt --lcc=1 --lambda-conv=20 --lambda-dense=42 --drop-conv=0.38 --drop-dense=0.36
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l2-dropout.txt --lcc=2 --lambda-conv=3 --lambda-dense=5.7 --drop-conv=0.15 --drop-dense=0.49
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/linf-dropout.txt --lcc=inf --lambda-conv=20 --lambda-dense=65 --drop-conv=0.16 --drop-dense=0.31

    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l1-batchnorm-dropout.txt --lcc=1 --lambda-conv=26 --lambda-dense=38 --lambda-bn=8.3 --batchnorm --drop-conv=0.3 --drop-dense=0.38
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/l2-batchnorm-dropout.txt --lcc=2 --lambda-conv=1.1 --lambda-dense=1.2 --lambda-bn=8.6 --batchnorm --drop-conv=0.14 --drop-dense=0.29
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=vgg --log-path=results/svhn/vgg/linf-batchnorm-dropout.txt --lcc=inf --lambda-conv=25 --lambda-dense=23 --lambda-bn=10 --batchnorm --drop-conv=0.34 --drop-dense=0.45
#done
