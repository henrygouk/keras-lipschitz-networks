#!/usr/bin/env python

from hyperopt import hp, fmin, tpe
import getopt
from sys import argv

batchnorm = False
dropout = False
lcc = None
spectral_decay = False
model_script = None

opts, args = getopt.getopt(argv[1:], "", ["batchnorm", "dropout", "lcc=", "spectral-decay", "model-script="])

for (k, v) in opts:
    if k == "--batchnorm":
        batchnorm = True
    elif k == "--dropout":
        dropout = True
    elif k == "--lcc":
        lcc = float(v)
    elif k == "--spectral-decay":
        spectral_decay = True
    elif k == "--model-script":
        model_script = v

def run_trial(cmd):
    import re
    import subprocess

    print "Executing", cmd
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in proc.stdout:
        if line.startswith("accuracy=") and re.match("^\d+?\.\d+?$", line[9:]):
            return 1.0 - float(line[9:])

    return 1

def model(lambda_conv=float("inf"), lambda_dense=float("inf"), lambda_bn=float("inf"), drop_conv=0, drop_dense=0, sd_conv=0, sd_dense=0):
    cmd = [model_script, "--valid"]

    if lcc == 1.0 or lcc == 2.0 or lcc == float("inf"):
        # LCC
        cmd.append("--lcc=" + str(lcc))
        cmd.append("--lambda-conv=" + str(lambda_conv))
        cmd.append("--lambda-dense=" + str(lambda_dense))

        if batchnorm:
            cmd.append("--lambda-bn=" + str(lambda_bn))

    if dropout:
        # Dropout
        cmd.append("--drop-conv=" + str(drop_conv))
        cmd.append("--drop-dense=" + str(drop_dense))

    if spectral_decay:
        # Spectral decay
        cmd.append("--sd-conv=" + str(sd_conv))
        cmd.append("--sd-dense=" + str(sd_dense))

    if batchnorm:
        cmd.append("--batchnorm")

    result = run_trial(cmd)
    print "result=%f" % result

    return result

def model_wrapper(kwargs):
    return model(**kwargs)

space = {}

if lcc == 2:
    space["lambda_conv"] = hp.lognormal("lambda_conv", 1.0, 0.5)
    space["lambda_dense"] = hp.lognormal("lambda_dense", 1.0, 0.5)

    if batchnorm:
        space["lambda_bn"] = hp.lognormal("lambda_bn", 1.0, 0.5)

if dropout:
    space["drop_conv"] = hp.uniform("drop_conv", 0.1, 0.5)
    space["drop_dense"] = hp.uniform("drop_dense", 0.1, 0.5)

best = fmin(fn=model_wrapper, space=space, algo=tpe.suggest, max_evals=20)

print best
