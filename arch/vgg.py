import keras
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D, InputSpec, Dropout
from keras.constraints import Constraint
from keras.regularizers import Regularizer
from maybe import maybe_batchnorm, maybe_dropout
from lipschitz import lcc_conv, lcc_dense, SpectralDecay

def vgg19(in_chan, in_dim, num_classes, **kwargs):
    sizes = [64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1]

    return vgg(in_chan, in_dim, num_classes, sizes, **kwargs)

def vgg_lite(in_chan, in_dim, num_classes, **kwargs):
    sizes = [64, 64, -1, 128, 128, -1, 192, 192, -1, 256, 256, -1]

    return vgg(in_chan, in_dim, num_classes, sizes, **kwargs)

def vgg(in_chan, in_dim, num_classes, sizes, bn=False, drop_rate_conv=0, drop_rate_dense=0, lcc_norm=2, lambda_conv=float("inf"), lambda_bn=float("inf"), lambda_dense=float("inf"), sd_conv=0, sd_dense=0):
    conv_reg = SpectralDecay(sd_conv)
    dense_reg = SpectralDecay(sd_dense)
    first = True

    model = Sequential()

    for s in sizes:
        if s == -1:
            # Pooling
            in_dim = in_dim / 2
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            # Conv layer
            if first:
                model.add(Conv2D(s, (3, 3), input_shape=(in_dim, in_dim, in_chan), kernel_regularizer=conv_reg, **lcc_conv(lcc_norm, lambda_conv, in_shape=(in_chan, in_dim, in_dim))))
                first = False
            else:
                model.add(Conv2D(s, (3, 3), kernel_regularizer=conv_reg, **lcc_conv(lcc_norm, lambda_conv, in_shape=(in_chan, in_dim, in_dim))))
            
            maybe_batchnorm(model, lambda_bn, bn)
            model.add(Activation('relu'))
            in_chan = s
    
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    model.add(Activation('relu'))
    maybe_dropout(model, drop_rate_dense)
    model.add(Dense(512, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    model.add(Activation('relu'))
    maybe_dropout(model, drop_rate_dense)
    model.add(Dense(num_classes, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    model.add(Activation('softmax'))

    return model