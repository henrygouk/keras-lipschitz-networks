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
epochs = 60
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
log_path = "/dev/null"
fold = 0

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
    "depth=",
    "log-path=",
    "fold="
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
    elif k == "--log-path":
        log_path = v
    elif k == "--fold":
        fold = int(v)

X_file = open(dataset + "/X.bin")
X_file.seek(fold * 10000 * 96 * 96 * 3, os.SEEK_SET)
y_file = open(dataset + "/y.bin")
y_file.seek(fold * 10000, os.SEEK_SET)
X_data = np.moveaxis(np.reshape(np.fromfile(X_file, dtype=np.ubyte, count=10000*96*96*3), (10000, 3, 96, 96)), [0, 1, 2, 3], [0, 3, 1, 2]).astype(np.float32)
y_data = np.reshape(np.fromfile(y_file, dtype=np.ubyte, count=10000), (10000, 1)).astype(np.float32)

x_train = X_data[0:9000]
y_train = y_data[0:9000]
x_test = X_data[9000:]
y_test = y_data[9000:]

if valid:
    x_test = x_train[0:1000]
    y_test = y_train[0:1000]
    x_train = x_train[1000:]
    y_train = y_train[1000:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

in_chan = x_train.shape[3]
in_dim = x_train.shape[1]

def lr_schedule_vgg(epoch):
    if epoch >= 50:
        return 0.000001
    elif epoch >= 40:
        return 0.00001
    else:
        return 0.0001

def lr_schedule_wrn(epoch):
    if epoch >= 80:
        return 0.0008
    elif epoch >= 60:
        return 0.004
    elif epoch >= 30:
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
    epochs = 60
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
    epochs = 100
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

with open(log_path, "a") as f:
    f.write("loss=" + str(scores[0]) + ",accuracy=" + str(scores[1]) + "\n")
