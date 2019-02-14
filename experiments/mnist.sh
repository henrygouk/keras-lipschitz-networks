#!/bin/bash

for i in $(seq 1 5)
do
    ./mnist.py --log-path=results/mnist/none.txt
    ./mnist.py --log-path=results/mnist/sd.txt --sd-conv=0.001 --sd-dense=0.001
    ./mnist.py --log-path=results/mnist/batchnorm.txt --batchnorm
    ./mnist.py --log-path=results/mnist/dropout.txt --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --log-path=results/mnist/batchnorm-dropout.txt --batchnorm --drop-conv=0.2 --drop-dense=0.5

    ./mnist.py --log-path=results/mnist/l1.txt --lcc=1 --lambda-conv=32 --lambda-dense=32
    ./mnist.py --log-path=results/mnist/l2.txt --lcc=2 --lambda-conv=8 --lambda-dense=8
    ./mnist.py --log-path=results/mnist/linf.txt --lcc=inf --lambda-conv=16 --lambda-dense=16

    ./mnist.py --log-path=results/mnist/l1-batchnorm.txt --lcc=1 --lambda-conv=4 --lambda-dense=4 --lambda-bn=16 --batchnorm
    ./mnist.py --log-path=results/mnist/l2-batchnorm.txt --lcc=2 --lambda-conv=1 --lambda-dense=1 --lambda-bn=16 --batchnorm
    ./mnist.py --log-path=results/mnist/linf-batchnorm.txt --lcc=inf --lambda-conv=3 --lambda-dense=3 --lambda-bn=16 --batchnorm

    ./mnist.py --log-path=results/mnist/l1-dropout.txt --lcc=1 --lambda-conv=40 --lambda-dense=40 --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --log-path=results/mnist/l2-dropout.txt --lcc=2 --lambda-conv=8 --lambda-dense=8 --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --log-path=results/mnist/linf-dropout.txt --lcc=inf --lambda-conv=20 --lambda-dense=20 --drop-conv=0.2 --drop-dense=0.5

    ./mnist.py --log-path=results/mnist/l1-batchnorm-dropout.txt --lcc=1 --lambda-conv=4 --lambda-dense=4 --lambda-bn=16 --batchnorm --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --log-path=results/mnist/l2-batchnorm-dropout.txt --lcc=2 --lambda-conv=1 --lambda-dense=1 --lambda-bn=16 --batchnorm --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --log-path=results/mnist/linf-batchnorm-dropout.txt --lcc=inf --lambda-conv=3 --lambda-dense=3 --lambda-bn=16 --batchnorm --drop-conv=0.2 --drop-dense=0.5
done
