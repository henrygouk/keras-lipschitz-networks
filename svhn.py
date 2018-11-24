#!/usr/bin/env python

from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10, cifar100
from keras.optimizers import adam, sgd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from arch.vgg import vgg_lite
from arch.wrn import wrn
from scipy import io
import numpy as np
import getopt
import os
from sys import argv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 100
num_classes = 10
epochs = 20
data_augmentation = True
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
width=4
depth=16
arch="wrn"

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
    "model-path=",
    "arch=",
    "width=",
    "depth="
])

for (k, v) in opts:
    if k == "--dataset":
        dataset = v
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
    elif k == "--width":
        width = int(v)
    elif k == "--depth":
        depth = int(v)
    elif k == "--arch":
        arch = v

train_data = io.loadmat(dataset + "/train_32x32.mat")
extra_data = io.loadmat(dataset + "/extra_32x32.mat")
test_data = io.loadmat(dataset + "/test_32x32.mat")

x_train = np.append(np.transpose(train_data["X"], (3, 0, 1, 2)), np.transpose(extra_data["X"], (3, 0, 1, 2)), axis=0)
y_train = np.append(train_data["y"], extra_data["y"])
x_test = np.transpose(test_data["X"], (3, 0, 1, 2))
y_test = test_data["y"]

if valid:
    x_test = x_train[0:10000]
    y_test = y_train[0:10000]
    x_train = x_train[10000:]
    y_train = y_train[10000:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train - 1, num_classes)
y_test = to_categorical(y_test - 1, num_classes)

in_chan = x_train.shape[3]
in_dim = x_train.shape[1]

def lr_schedule_vgg(epoch):
    if epoch >= 18:
        return 0.000001
    elif epoch >= 15:
        return 0.00001
    else:
        return 0.0001

def lr_schedule_wrn(epoch):
    if epoch >= 16:
        return 0.0008
    elif epoch >= 12:
        return 0.004
    elif epoch >= 6:
        return 0.02
    else:
        return 0.1

if arch == "vgg":
    model = vgg_lite(
        in_chan,
        in_dim,
        num_classes,
        bn=batchnorm,
        drop_rate_conv=drop_conv,
        drop_rate_dense=drop_dense,
        lcc_norm=lcc_norm,
        lambda_conv=lambda_conv,
        lambda_dense=lambda_dense,
        lambda_bn=lambda_bn,
        sd_conv=sd_conv,
        sd_dense=sd_dense
    )

    lr_scheduler = LearningRateScheduler(lr_schedule_vgg)
    opt = adam(amsgrad=True)
elif arch == "wrn":
    model = wrn(
        in_chan,
        in_dim,
        num_classes,
        width,
        depth,
        drop_rate_conv=drop_conv,
        lcc_norm=lcc_norm,
        lambda_conv=lambda_conv,
        lambda_dense=lambda_dense,
        lambda_bn=lambda_bn,
        sd_conv=sd_conv,
        sd_dense=sd_dense
    )

    batch_size = 50
    lr_scheduler = LearningRateScheduler(lr_schedule_wrn)
    opt = sgd(momentum=0.9, nesterov=True)
else:
    raise Exception("Unknown architecture")

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 128
x_test /= 128
x_train -= 1
x_test -= 1

datagen = ImageDataGenerator(
    width_shift_range=0.125,
    height_shift_range=0.125,
    fill_mode='nearest',
    horizontal_flip=True)

datagen.fit(x_train)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[lr_scheduler])

model.save(model_path)

scores = model.evaluate(x_test, y_test, verbose=1)
print 'loss=%f' % scores[0]
print 'accuracy=%f' % scores[1]
