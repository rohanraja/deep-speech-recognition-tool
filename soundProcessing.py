import scipy.io.wavfile as wv
from config import config

from module import getModule
from threading import Thread
import numpy as np
import cv2, os
from Queue import Queue

def processImage(img):

    return img


def processWav(wavPath):
    global i
    
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

CHUNKSIZE = 1024*5
import hashlib

def getImageCacheFileName():

    infos = "%s_%s_%s_%s_%s" % (
            config["height"], 
            config["width"],
            config["channels"],
            config["wavToImage"],
            config["wavVariations"],
    )
    hashVal = hashlib.sha224(infos).hexdigest()
    cacheFile = os.path.join("imageCaches", "%s.npy" % hashVal)
    return cacheFile

class DataGenerationThread(Thread):

    def __init__(self, wpaths):
        super(DataGenerationThread, self).__init__()
        self.wavPaths = wpaths

    def run(self):
        global queue

        cacheFile = getImageCacheFileName()
        
        try:
            print "Trying to load cache file"
            cachedData = np.load(cacheFile)
            cachedData = cachedData[()]
            print "Loaded Image cache"
        except:
            cachedData = {}

        chunk = []
        while True:
            for wavPath in self.wavPaths:
                try:
                    label = getLabel(wavPath)
                    if wavPath in cachedData.keys():
                        imgs = cachedData[wavPath]
                    else:
                        imgs = processWav(wavPath)
                        cachedData[wavPath] = imgs

                    for img in imgs:
                        chunk.append((img, label))
                        if len(chunk) == CHUNKSIZE:
                            queue.put(chunk)
                            chunk = []

                except Exception, e:
                    # print "Error: %s\nprocessing file %s" % (e, wavPath)
                    pass

            if os.path.isfile(cacheFile) == False:
                np.save(cacheFile, cachedData)





from keras.utils import np_utils

IMG_WIDTH = int(config["width"])
IMG_HEIGHT = int(config["height"])

def make_trainable(val):

    Y = np_utils.to_categorical([val[1]], int(config["classes"]) )[0]
    img = val[0]
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_LINEAR)
    img = img.reshape((img.shape[0], img.shape[1], int(config["channels"]) )).astype("float32")
    X = np.rollaxis(img, 2)
    X = X/ 255.0

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
        

        vals = queue.get()
        queue.task_done()
        random.shuffle(vals)
        
        j = 0

        while j < len(vals):
            X_train = []
            Y_train = []
            for i in range(bsize):

                val = vals[j]
                j+=1
                X, Y = make_trainable(val)
                X_train.append(X)
                Y_train.append(Y)


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


