from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
# from keras.layers import LocallyConnected2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from config import config
import sys
from googlenet import create_googlenet

sys.setrecursionlimit(10000)


nb_classes = config["classes"]
img_rows = config["width"]
img_cols = config["height"]
img_channels = config["channels"]


class MODELS():
    def local1(self, params):
        model = Sequential()

        model.add(LocallyConnected2D(32, 7, 7, 
                                input_shape=(img_channels, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        model.summary()
        return model

    def ms_model(self, params):
        model = Sequential()

        model.add(Convolution2D(150, 8, 8, 
            border_mode='same',
            subsample=(2,2),
            input_shape=(img_channels, img_rows, img_cols)
            )
        )
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(6, 6)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        model.summary()
        return model
    def convnet1(self, params):
        model = Sequential()


        model.add(Convolution2D(32, 3, 3, border_mode='same',
                                 input_shape=(img_channels, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(32, 1, 1, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        model.summary()
        return model


    def gnet(self, params):
        model = create_googlenet()
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        return model

def getModel():
    
    models = MODELS()
    modelName = config.get("model", "")
    modelParams = config.get("modelParams", {})
    retModel = getattr(models, modelName, models.convnet1)(modelParams)

    return retModel


