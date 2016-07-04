'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from keras import callbacks

batch_size = 64
nb_classes = 10
nb_epoch = 50
data_augmentation = True

# input image dimensions
# the CIFAR10 images are RGB
img_channels = 1

from model import getModel

model = getModel()

from datagen import gen_batch, numSamples, IMAGE_WIDTH, IMAGE_HEIGHT, get_test

img_rows = IMAGE_WIDTH
img_cols = IMAGE_HEIGHT

X_test, Y_test = get_test()

print(X_test.shape)
print("Got Test Data")


# model.load_weights("savedModel_21")
# out = model.evaluate(X_test, Y_test)


class SaveEpoch(callbacks.Callback):
    def on_epoch_end(self, epch, thrd = {}):

        model.save_weights("savedModel_%d"%epch, overwrite=True)



model.fit_generator(gen_batch(batch_size),
        samples_per_epoch=numSamples(batch_size),
        nb_epoch= nb_epoch,
        validation_data=(X_test, Y_test),
        max_q_size = 100,
        callbacks=[SaveEpoch()]
)


