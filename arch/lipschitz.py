import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D, InputSpec
from keras.constraints import Constraint
from keras.regularizers import Regularizer
from keras import backend as K

class SpectralDecay(Regularizer):
    
    def __init__(self, sd_lambda, iterations=2):
        self.sd_lambda = sd_lambda
        self.iterations = iterations
    
    def __call__(self, w):
        if self.sd_lambda == 0:
            return 0
        else:
            w = K.reshape(w, [-1, w.shape.as_list()[-1]])
            x = K.random_normal_variable(shape=(int(w.shape[1]), 1), mean=0, scale=1)

            for i in range(0, self.iterations): 
                x_p = K.dot(w, x)
                x = K.dot(K.transpose(w), x_p)
            
            return self.sd_lambda * K.sum(K.pow(K.dot(w, x), 2.0)) / K.sum(K.pow(x, 2.0))
    
    def get_config(self):
        return {
            "sd_lambda": self.sd_lambda,
            "iterations": self.iterations
        }

def lcc_conv(lcc_norm, lcc_lambda, in_shape=None, stride=(1,1), padding="same", iterations=2):
    const = None
    
    if lcc_norm == 1:
        const = L1Lipschitz(lcc_lambda)
    elif lcc_norm == 2:
        const = L2Lipschitz(lcc_lambda, in_shape, stride, padding, iterations)
    elif lcc_norm == float("inf"):
        const = LInfLipschitz(lcc_lambda)
    else:
        raise Exception("Unknown LCC norm")

    return {
        "kernel_constraint": const,
        "strides": stride,
        "padding": padding
    }

def lcc_dense(lcc_norm, lcc_lambda, iterations=2):
    const = None
    
    if lcc_norm == 1:
        const = L1Lipschitz(lcc_lambda)
    elif lcc_norm == 2:
        const = L2Lipschitz(lcc_lambda, in_shape=None, iterations=iterations)
    elif lcc_norm == float("inf"):
        const = LInfLipschitz(lcc_lambda)
    else:
        raise Exception("Unknown LCC norm")

    return {
        "kernel_constraint": const
    }

def lcc_batchnorm(lcc_lambda):
    return BNLipschitz(lcc_lambda)

class L1Lipschitz(Constraint):

    def __init__(self, max_k):
        self.max_k = max_k

    def __call__(self, w):
        axes=1

        if len(w.shape) == 4:
            axes=[0, 1, 3]
        
        norm = K.max(K.sum(K.abs(w), axis=axes, keepdims=False))
        return w * (1.0 / K.maximum(1.0, norm / self.max_k))
    
    def get_config(self):
        return {"max_k": self.max_k}

class L2Lipschitz(Constraint):

    def __init__(self, max_k, in_shape, stride=(1,1), padding="same", iterations=2):
        self.max_k = max_k
        self.in_shape = in_shape
        self.stride = stride
        self.padding = padding
        self.iterations = iterations
    
    def __call__(self, w):
        norm = self.max_k

        if len(w.shape) == 4:
            x = K.random_normal_variable(shape=(1,) + self.in_shape[1:3] + (self.in_shape[0],), mean=0, scale=1)

            for i in range(0, self.iterations): 
                x_p = K.conv2d(x, w, strides=self.stride, padding=self.padding)
                x = K.conv2d_transpose(x_p, w, x.shape, strides=self.stride, padding=self.padding)
            
            Wx = K.conv2d(x, w, strides=self.stride, padding=self.padding)
            norm = K.sqrt(K.sum(K.pow(Wx, 2.0)) / K.sum(K.pow(x, 2.0)))
        else:
            x = K.random_normal_variable(shape=(int(w.shape[1]), 1), mean=0, scale=1)

            for i in range(0, self.iterations): 
                x_p = K.dot(w, x)
                x = K.dot(K.transpose(w), x_p)
            
            norm = K.sqrt(K.sum(K.pow(K.dot(w, x), 2.0)) / K.sum(K.pow(x, 2.0)))

        return w * (1.0 / K.maximum(1.0, norm / self.max_k))
    
    def get_config(self):
        return {
            "max_k": self.max_k,
            "in_shape": self.in_shape,
            "stride": self.stride,
            "padding": self.padding,
            "iterations": self.iterations
        }

class LInfLipschitz(Constraint):

    def __init__(self, max_k):
        self.max_k = max_k

    def __call__(self, w):
        axes=0

        if len(w.shape) == 4:
            axes=[0, 1, 2]
        
        norm = K.max(K.sum(K.abs(w), axis=axes, keepdims=False))
        return w * (1.0 / K.maximum(1.0, norm / self.max_k))
    
    def get_config(self):
        return {"max_k": self.max_k}

class BNConstraint(Constraint):

    def __init__(self, max_k, moving_variance):
        self.max_k = max_k
        self.moving_variance = moving_variance
    
    def __call__(self, w):
        norm = K.max(K.abs(w / K.sqrt(self.moving_variance + 1e-6)))
        return w * (1.0 / K.maximum(1.0, norm / self.max_k))
    
    def get_config(self):
        return {"max_k": self.max_k}

class BNLipschitz(BatchNormalization):

    def __init__(self, max_k=float("inf"), **kwargs):
        self.max_k = max_k
        super(BNLipschitz, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)

        gamma_constraint = BNConstraint(self.max_k, self.moving_variance)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        self.built = True
