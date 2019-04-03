#!/bin/bash
#i=$1
for i in $(seq 0 9)
do
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/none.txt
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/batchnorm.txt --batchnorm
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/sd.txt --sd-conv=0.001 --sd-dense=0.01
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/dropout.txt --drop-conv=0.24 --drop-dense=0.47
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/batchnorm-dropout.txt --drop-conv=0.16 --drop-dense=0.44 --batchnorm

#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l1.txt --lcc=1 --lambda-conv=46 --lambda-dense=67
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l2.txt --lcc=2 --lambda-conv=2.8 --lambda-dense=7.9
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/linf.txt --lcc=inf --lambda-conv=36.5 --lambda-dense=60.5

    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l1-batchnorm.txt --lcc=1 --lambda-conv=26 --lambda-dense=89 --lambda-bn=3 --batchnorm
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l2-batchnorm.txt --lcc=2 --lambda-conv=8.4 --lambda-dense=7.3 --lambda-bn=2.3 --batchnorm
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/linf-batchnorm.txt --lcc=inf --lambda-conv=22 --lambda-dense=53 --lambda-bn=2.5 --batchnorm

#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l1-dropout.txt --lcc=1 --lambda-conv=65 --lambda-dense=61 --drop-conv=0.48 --drop-dense=0.5
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l2-dropout.txt --lcc=2 --lambda-conv=5.1 --lambda-dense=6.5 --drop-conv=0.28 --drop-dense=0.41
#    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/linf-dropout.txt --lcc=inf --lambda-conv=50 --lambda-dense=56 --drop-conv=0.46 --drop-dense=0.44

    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l1-batchnorm-dropout.txt --lcc=1 --lambda-conv=20 --lambda-dense=42 --lambda-bn=3.7 --drop-conv=0.16 --drop-dense=0.31 --batchnorm
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/l2-batchnorm-dropout.txt --lcc=2 --lambda-conv=1.2 --lambda-dense=3.8 --lambda-bn=6.3 --drop-conv=0.34 --drop-dense=0.5 --batchnorm
    ./sins10.py --fold=$i --dataset=/research/repository/hgrg1/sins10 --arch=vgg --log-path=results/sins10/vgg/linf-batchnorm-dropout.txt --lcc=inf --lambda-conv=22 --lambda-dense=30 --lambda-bn=5.2 --drop-conv=0.35 --drop-dense=0.29 --batchnorm
done
