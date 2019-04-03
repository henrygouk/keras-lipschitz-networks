#!/bin/bash

for i in $(seq 0 9)
do
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/none.txt
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/sd.txt --sd-conv=0.01 --sd-dense=0.1
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/dropout.txt --drop-conv=0.45

    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/l1.txt --lcc=1 --lambda-conv=21 --lambda-dense=70 --lambda-bn=12
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/l2.txt --lcc=2 --lambda-conv=5.8 --lambda-dense=1.2 --lambda-bn=9.9
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/linf.txt --lcc=inf --lambda-conv=59 --lambda-dense=37 --lambda-bn=11

    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/l1-dropout.txt --lcc=1 --lambda-conv=21 --lambda-dense=33 --lambda-bn=5.5 --drop-conv=0.36
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/l2-dropout.txt --lcc=2 --lambda-conv=4.9 --lambda-dense=2.4 --lambda-bn=2.9 --drop-conv=0.47
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=wrn --log-path=results/sins10/wrn/linf-dropout.txt --lcc=inf --lambda-conv=20 --lambda-dense=48 --lambda-bn=10 --drop-conv=0.36
done
