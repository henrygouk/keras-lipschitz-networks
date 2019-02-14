#!/bin/bash

for i in $(seq 1 5)
do
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/none.txt
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/sd.txt --sd-conv=0.001 --sd-dense=0.001
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/batchnorm.txt --batchnorm
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/dropout.txt --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/batchnorm-dropout.txt --batchnorm --drop-conv=0.2 --drop-dense=0.5

    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l1.txt --lcc=1 --lambda-conv=48 --lambda-dense=48
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l2.txt --lcc=2 --lambda-conv=8 --lambda-dense=8
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/linf.txt --lcc=inf --lambda-conv=36 --lambda-dense=36

    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l1-batchnorm.txt --lcc=1 --lambda-conv=6 --lambda-dense=6 --lambda-bn=20 --batchnorm
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l2-batchnorm.txt --lcc=2 --lambda-conv=2 --lambda-dense=2 --lambda-bn=8 --batchnorm
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/linf-batchnorm.txt --lcc=inf --lambda-conv=6 --lambda-dense=6 --lambda-bn=20 --batchnorm

    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l1-dropout.txt --lcc=1 --lambda-conv=56 --lambda-dense=56 --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l2-dropout.txt --lcc=2 --lambda-conv=12 --lambda-dense=12 --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/linf-dropout.txt --lcc=inf --lambda-conv=48 --lambda-dense=48 --drop-conv=0.2 --drop-dense=0.5

    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l1-batchnorm-dropout.txt --lcc=1 --lambda-conv=12 --lambda-dense=12 --lambda-bn=20 --batchnorm --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/l2-batchnorm-dropout.txt --lcc=2 --lambda-conv=4 --lambda-dense=4 --lambda-bn=8 --batchnorm --drop-conv=0.2 --drop-dense=0.5
    ./mnist.py --dataset=fashion-mnist --log-path=results/fashion-mnist/linf-batchnorm-dropout.txt --lcc=inf --lambda-conv=12 --lambda-dense=12 --lambda-bn=20 --batchnorm --drop-conv=0.2 --drop-dense=0.5
done
