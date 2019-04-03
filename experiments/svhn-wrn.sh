#!/bin/bash

#for i in $(seq 1 5)
#do
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/none.txt
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/sd.txt --sd-conv=0.01 --sd-dense=0.1
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/dropout.txt --drop-conv=0.48

    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/l1.txt --lcc=1 --lambda-conv=89 --lambda-dense=34 --lambda-bn=8.5
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/l2.txt --lcc=2 --lambda-conv=5.6 --lambda-dense=5.6 --lambda-bn=9.3
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/linf.txt --lcc=inf --lambda-conv=58 --lambda-dense=63 --lambda-bn=7.5

    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/l1-dropout.txt --lcc=1 --lambda-conv=56 --lambda-dense=48 --lambda-bn=5.8 --drop-conv=0.39
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/l2-dropout.txt --lcc=2 --lambda-conv=5.2 --lambda-dense=8.7 --lambda-bn=6.3 --drop-conv=0.45
    ./svhn.py --dataset=/research/repository/hgrg1/svhn --arch=wrn --log-path=results/svhn/wrn/linf-dropout.txt --lcc=inf --lambda-conv=82 --lambda-dense=46 --lambda-bn=6.2 --drop-conv=0.43
#done