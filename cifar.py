#!/usr/bin/env python

from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10, cifar100
from keras.optimizers import adam, sgd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from arch.vgg19 import vgg19
from arch.wrn import wrn
import getopt
import os
from sys import argv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 100
num_classes = 0
epochs = 200
data_augmentation = True
(x_train, y_train), (x_test, y_test) = ((None, None), (None, None))
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
width=10
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
        if v == "cifar10":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            num_classes = 10
        elif v == "cifar100":
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            num_classes = 100
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
    elif k == "--width":
        width = int(v)
    elif k == "--depth":
        depth = int(v)
    elif k == "--arch":
        arch = v

if valid:
    x_test = x_train[40000:]
    y_test = y_train[40000:]
    x_train = x_train[0:40000]
    y_train = y_train[0:40000]

# The data, split between train and test sets:
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

in_chan = x_train.shape[3]
in_dim = x_train.shape[1]

model = None
lr_scheduler = None
opt = None

def lr_schedule_vgg(epoch):
    if epoch >= 120:
        return 0.000001
    elif epoch >= 100:
        return 0.00001
    else:
        return 0.0001

def lr_schedule_wrn(epoch):
    if epoch >= 160:
        return 0.0001
    elif epoch >= 120:
        return 0.001
    elif epoch >= 60:
        return 0.01
    else:
        return 0.1

if arch == "vgg":
    model = vgg19(
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

    epochs = 140
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

    epochs = 200
    lr_scheduler = LearningRateScheduler(lr_schedule_wrn)
    opt = sgd(decay=5e-4, momentum=0.9, nesterov=True)
else:
    raise Exception("Unknown architecture")

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 128
x_test /= 128
x_train -= 1
x_test -= 1

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    fill_mode='nearest',  # set mode for filling points outside the input boundaries
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    rescale=None,  # set rescaling factor (applied before any other transformation)
    preprocessing_function=None,  # set function that will be applied on each input
    data_format=None,  # image data format, either "channels_first" or "channels_last"
    validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train,
                                    batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=4,
                    callbacks=[lr_scheduler],
                    steps_per_epoch=x_train.shape[0] / batch_size)

# Save model and weights
model.save(model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print 'loss=%f' % scores[0]
print 'accuracy=%f' % scores[1]
