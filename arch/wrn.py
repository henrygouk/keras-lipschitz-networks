import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, Dropout, Input
from keras.layers.merge import add
from keras.models import Model
from keras.regularizers import l2
from maybe import maybe_dropout
from lipschitz import lcc_batchnorm, lcc_conv, lcc_dense, SpectralDecay

def wrn(in_chan, in_dim, num_classes, k, d, drop_rate_conv=0, lcc_norm=2, lambda_conv=float("inf"), lambda_bn=float("inf"), lambda_dense=float("inf"), sd_conv=0, sd_dense=0):
    conv_reg = l2(0.00005)
    dense_reg = l2(0.00005)
    
    if sd_conv != 0:
        conv_reg = SpectralDecay(sd_conv)
    
    if sd_dense != 0:
        dense_reg = SpectralDecay(sd_dense)

    blocks_per_group = (d - 4) / 6
    widening_factor = k

    inputs = Input(shape=(in_dim, in_dim, in_chan))

    x = Conv2D(16, (3, 3), use_bias=False, kernel_regularizer=conv_reg, **lcc_conv(lcc_norm, lambda_conv, in_shape=(in_chan, in_dim, in_dim)))(inputs)

    for i in range(0, blocks_per_group):
        nb_filters = 16 * widening_factor
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=1, drop_prob=drop_rate_conv, lcc_norm=lcc_norm, lambda_conv=lambda_conv, lambda_bn=lambda_bn, conv_reg=conv_reg)

    for i in range(0, blocks_per_group):
        nb_filters = 32 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor, drop_prob=drop_rate_conv, lcc_norm=lcc_norm, lambda_conv=lambda_conv, lambda_bn=lambda_bn, conv_reg=conv_reg)

    for i in range(0, blocks_per_group):
        nb_filters = 64 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor, drop_prob=drop_rate_conv, lcc_norm=lcc_norm, lambda_conv=lambda_conv, lambda_bn=lambda_bn, conv_reg=conv_reg)

    x = lcc_batchnorm(lambda_bn)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(8, 8), strides=None, padding="valid")(x)
    x = Flatten()(x)

    x = Dense(num_classes, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense))(x)
    predictions = Activation("softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model

def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def residual_block(x, nb_filters=16, subsample_factor=1, drop_prob=0.0, lcc_norm=2, lambda_conv=float("inf"), lambda_bn=float("inf"), conv_reg=None):
    
    prev_shape = K.int_shape(x)
    prev_nb_channels = prev_shape[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        shortcut = AveragePooling2D(pool_size=subsample)(x)
    else:
        subsample = (1, 1)
        shortcut = x
        
    if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

    y = lcc_batchnorm(lambda_bn)(x)
    y = Activation('relu')(y)
    y = Conv2D(nb_filters, (3, 3), use_bias=False, kernel_regularizer=conv_reg, **lcc_conv(lcc_norm, lambda_conv, in_shape=(prev_nb_channels, prev_shape[1], prev_shape[2]), stride=subsample))(y)
    y = lcc_batchnorm(lambda_bn)(y)
    y = Activation('relu')(y)

    if drop_prob != 0.0:
        y = Dropout(drop_prob)(y)
    
    y = Conv2D(nb_filters, (3, 3), use_bias=False, kernel_regularizer=conv_reg, **lcc_conv(lcc_norm, lambda_conv, in_shape=(nb_filters, prev_shape[1], prev_shape[2])))(y)
    
    out = add([y, shortcut])

    return out