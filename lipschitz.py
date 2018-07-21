import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D, InputSpec
from keras.constraints import Constraint
from keras import backend as K


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
