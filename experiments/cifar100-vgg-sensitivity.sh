#!/bin/bash

# 0.8 * \lambda
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l1-batchnorm-0.8.txt --lcc=1 --lambda-conv=18.4 --lambda-dense=41.6 --lambda-bn=6.72 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l2-batchnorm-0.8.txt --lcc=2 --lambda-conv=1.52 --lambda-dense=4.88 --lambda-bn=4.56 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/linf-batchnorm-0.8.txt --lcc=inf --lambda-conv=16 --lambda-dense=44.8 --lambda-bn=3.76 --batchnorm

# 1.0 * \lambda
# ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l1-batchnorm.txt --lcc=1 --lambda-conv=23 --lambda-dense=52 --lambda-bn=8.4 --batchnorm
# ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/l2-batchnorm.txt --lcc=2 --lambda-conv=1.9 --lambda-dense=6.1 --lambda-bn=5.7 --batchnorm
# ./cifar100.py --arch=vgg --log-path=results/cifar100/vgg/linf-batchnorm.txt --lcc=inf --lambda-conv=20 --lambda-dense=56 --lambda-bn=4.7 --batchnorm

# 1.2 * \lambda
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l1-batchnorm-1.2.txt --lcc=1 --lambda-conv=27.6 --lambda-dense=62.4 --lambda-bn=10.08 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l2-batchnorm-1.2.txt --lcc=2 --lambda-conv=2.28 --lambda-dense=7.32 --lambda-bn=6.84 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/linf-batchnorm-1.2.txt --lcc=inf --lambda-conv=24 --lambda-dense=67.2 --lambda-bn=5.64 --batchnorm

# 1.4 * \lambda
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l1-batchnorm-1.4.txt --lcc=1 --lambda-conv=32.2 --lambda-dense=72.8 --lambda-bn=11.76 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l2-batchnorm-1.4.txt --lcc=2 --lambda-conv=2.66 --lambda-dense=8.54 --lambda-bn=7.98 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/linf-batchnorm-1.4.txt --lcc=inf --lambda-conv=28 --lambda-dense=78.4 --lambda-bn=6.58 --batchnorm

# 1.6 * \lambda
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l1-batchnorm-1.6.txt --lcc=1 --lambda-conv=36.8 --lambda-dense=83.2 --lambda-bn=13.44 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l2-batchnorm-1.6.txt --lcc=2 --lambda-conv=3.04 --lambda-dense=9.76 --lambda-bn=9.12 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/linf-batchnorm-1.6.txt --lcc=inf --lambda-conv=32 --lambda-dense=89.6 --lambda-bn=7.52 --batchnorm

# 1.8 * \lambda
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l1-batchnorm-1.8.txt --lcc=1 --lambda-conv=41.4 --lambda-dense=93.6 --lambda-bn=15.12 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/l2-batchnorm-1.8.txt --lcc=2 --lambda-conv=3.42 --lambda-dense=10.98 --lambda-bn=10.26 --batchnorm
./cifar100.py --arch=vgg --log-path=results/cifar100/sensitivity/linf-batchnorm-1.8.txt --lcc=inf --lambda-conv=36 --lambda-dense=100.80 --lambda-bn=8.46 --batchnorm