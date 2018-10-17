from keras.layers import Dropout
from lipschitz import lcc_batchnorm

def maybe_dropout(model, drop_rate):
    if drop_rate != 0:
        model.add(Dropout(drop_rate))

def maybe_batchnorm(model, lcc_lambda, do_bn):
    if do_bn:
        model.add(lcc_batchnorm(lcc_lambda))
