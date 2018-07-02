from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, BatchNormalization, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import Constraint
from keras import backend as K
import os

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

class BNLipschitz(Constraint):

    def __init__(self, max_k):
        self.max_k = max_k
    
    def __call__(self, w):
        norm = K.max(K.sum(K.abs(w)))
        return w * (1.0 / K.maximum(1.0, norm / self.max_k))
    
    def get_config(self):
        return {"max_k": self.max_k}


batch_size = 100
num_classes = 10
epochs = 140
data_augmentation = True
lcc_l1 = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

const = None
bnconst = None

if lcc_l1:
    const = L1Lipschitz(20.0)
    bnconst = BNLipschitz(20.0)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:], kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_constraint=const))
model.add(BatchNormalization(gamma_constraint=bnconst))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, kernel_constraint=const))
model.add(Activation('relu'))
model.add(Dense(num_classes, kernel_constraint=const))
model.add(Activation('softmax'))

def lr_schedule(epoch):
    lr = 0.0001

    if epoch >= 100:
        lr = 0.00001
    elif epoch >= 120:
        lr = 0.000001

    return lr

# initiate optimizer
opt = keras.optimizers.adam(lr=lr_schedule(0), decay=1e-6)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

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
                    callbacks=[lr_scheduler])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
