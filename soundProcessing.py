import scipy.io.wavfile as wv
from config import config

from module import getModule
from threading import Thread
import numpy as np
import cv2, os
from Queue import Queue



def processImage(img):

    return img
    img = getModule("Resize")(img)
    img = getModule("Mean")(img)




def processWav(wavPath):
    
    sampRate, wavData = wv.read(wavPath)

    variationsWav, sampRate = getModule("wavVariations")(wavData, sampRate)

    images = []

    for wData in variationsWav:
        wavImage = getModule("wavToImage")(wData, sampRate)

        processedImg = processImage(wavImage)
        images.append(processedImg)

    return images


def getLabel(fPath):
    return int(fPath.split("/")[-1].split("_")[0])

def generateImages(wavPaths):

    Xs = [] 
    for wavPath in wavPaths:
        label = getLabel(wavPath)
        for img in processWav(wavPath):
            Xs.append( (img, label) )


    return Xs


queue = Queue(400)

class DataGenerationThread(Thread):

    def __init__(self, wpaths):
        super(DataGenerationThread, self).__init__()
        self.wavPaths = wpaths

    def run(self):
        global queue

        while True:

            for wavPath in self.wavPaths:
                try:
                    label = getLabel(wavPath)
                    for img in processWav(wavPath):
                        queue.put((img, label))
                except:
                    print "Error processing file %s" % wavPath



from keras.utils import np_utils

IMG_WIDTH = int(config["width"])
IMG_HEIGHT = int(config["height"])

def make_trainable(val):

    Y = np_utils.to_categorical([val[1]], 10)

    img = val[0]
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)
    img = img.reshape((img.shape[0], img.shape[1], 1)).astype("float32")
    X = np.rollaxis(img, 2)

    return X, Y

DUMMY_WAV = "./dummy.wav"
def num_samples(wavs):
    return len(wavs) * len(processWav(DUMMY_WAV))


import random

def batch_generator(bsize, wavs):

    global queue
    dth = DataGenerationThread(wavs)
    dth.start()

    while True:

        X_train = []
        Y_train = []
        
        for i in range(bsize):

            val = queue.get()
            queue.task_done()

            X, Y = make_trainable(val)
            X_train.append(X)
            Y_train.append(Y)

        both = zip(X_train, Y_train)
        random.shuffle(both)
        X_train = zip(*both)[0]
        Y_train = zip(*both)[1]

        yield np.array(X_train, dtype="float32"), np.array(Y_train, dtype="int")



def batchgen_and_numsamples():
    wavFiles = []
    spoDir = config["wavDir"]
    for f in os.listdir(spoDir):
        if f.endswith('.wav'):
            wavFiles.append(os.path.join(spoDir, f))

    random.shuffle(wavFiles)
    batchgen = batch_generator(int(config["batch_size"]), wavFiles)
    numSamples = num_samples(wavFiles)

    return batchgen, numSamples


