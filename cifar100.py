#!/usr/bin/env python

import os
import sys

command = "./cifar.py --dataset=cifar100 " + " ".join(sys.argv[1:])
os.system(command)