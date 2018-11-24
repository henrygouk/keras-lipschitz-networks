#!/usr/bin/env python

from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist, fashion_mnist
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from arch.maybe import maybe_batchnorm, maybe_dropout
from arch.lipschitz import lcc_conv, lcc_dense, SpectralDecay
import getopt
import os
from sys import argv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 100
epochs = 20
num_classes = 10
lcc_norm = 2
lambda_conv = float("inf")
lambda_dense = float("inf")
lambda_bn = float("inf")
drop_conv = 0
drop_dense = 0
sd_conv=0
sd_dense=0
batchnorm = False
model_path = "/dev/null"
valid = False
img_rows, img_cols = 28, 28
loaded = False

opts, args = getopt.getopt(argv[1:], "", longopts=[
    "dataset=",
    "valid",
    "lcc=",
    "lambda-conv=",
    "lambda-dense=",
    "lambda-bn=",
    "drop-conv=",
    "drop-dense=",
    "sd-conv=",
    "sd-dense=",
    "batchnorm",
    "model-path="
])

for (k, v) in opts:
    if k == "--dataset":
        loaded = True
        if v == "mnist":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif v == "fashion-mnist":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            raise Exception("Unknown dataset")
    elif k == "--valid":
        valid = True
    elif k == "--lcc":
        lcc_norm = float(v)
    elif k == "--lambda-conv":
        lambda_conv = float(v)
    elif k == "--lambda-dense":
        lambda_dense = float(v)
    elif k == "--lambda-bn":
        lambda_bn = float(v)
    elif k == "--drop-conv":
        drop_conv = float(v)
    elif k == "--drop-dense":
        drop_dense = float(v)
    elif k == "--sd-conv":
        sd_conv = float(v)
    elif k == "--sd-dense":
        sd_dense = float(v)
    elif k == "--batchnorm":
        batchnorm = True
    elif k == "--model-path":
        model_path = v

if not loaded:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

if valid:
    x_test = x_train[50000:]
    y_test = y_train[50000:]
    x_train = x_train[0:50000]
    y_train = y_train[0:50000]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

in_chan = x_train.shape[3]
in_dim = x_train.shape[1]

def lr_schedule(epoch):
    return 0.0001

lr_scheduler = LearningRateScheduler(lr_schedule)
opt = adam(amsgrad=True)

conv_reg = SpectralDecay(sd_conv)
dense_reg = SpectralDecay(sd_dense)

model = Sequential()
model.add(Conv2D(64, (5, 5), kernel_regularizer=conv_reg, **lcc_conv(lcc_norm, lambda_conv, in_shape=(1, 28, 28))))
maybe_batchnorm(model, lambda_bn, batchnorm)
model.add(Activation("relu"))
maybe_dropout(model, drop_conv)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), kernel_regularizer=conv_reg, **lcc_conv(lcc_norm, lambda_conv, in_shape=(64, 14, 14))))
maybe_batchnorm(model, lambda_bn, batchnorm)
model.add(Activation("relu"))
maybe_dropout(model, drop_conv)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
maybe_batchnorm(model, lambda_bn, batchnorm)
model.add(Activation("relu"))
maybe_dropout(model, drop_dense)
model.add(Dense(num_classes, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 128
x_test /= 128
x_train -= 1
x_test -= 1

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[lr_scheduler])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
