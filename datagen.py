import os
import glob
import random
import numpy as np

import cv2
from keras.utils import np_utils


#Size of images
IMAGE_WIDTH = 13
IMAGE_HEIGHT = 198
# IMAGE_WIDTH = 224
# IMAGE_HEIGHT = 224
PERCENT_TRAIN = 0.9

meanImg = cv2.imread("mean.png", 0)

def transform_img(img, img_width=IMAGE_HEIGHT, img_height=IMAGE_WIDTH):

    #Histogram Equalization
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_LINEAR)
    img = img.reshape((img.shape[0], img.shape[1], 1)).astype("float32")

    img[:,:,0] = img[:,:,0].astype("float32") - meanImg.astype("float32")
    return img


nChannel = 1
train_data = [img for img in glob.glob("/mnt/2tb/datasets/spokensetMFCC/*/*.png")]
test_data = [img for img in glob.glob("/mnt/2tb/datasets/spokensetMFCCtest/*/*.png")]

MIX = False

if MIX:
    both_data = train_data + test_data
    random.shuffle(both_data)
    tot = len(both_data)
    nTrain = int(tot * PERCENT_TRAIN)
    nTest = tot - nTrain
    train_data = both_data[:nTrain]
    test_data = both_data[nTrain:]
else:
    nTrain = len(train_data)
    nTest = len(test_data)
    tot = nTrain + nTest
    random.shuffle(train_data)


def getLabel(fPath):
    return int(fPath.split("/")[-1].split("_")[0])


def processImg(img_path):

    img = cv2.imread(img_path, 0)
    img = transform_img(img, img_width=IMAGE_HEIGHT, img_height=IMAGE_WIDTH)
    label = getLabel(img_path)
    return np.rollaxis(img, 2)



def processBatchUrls(urls):

    bsize = len(urls)

    X_train = np.zeros((
            bsize,
            nChannel,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,

        ))

    Y_train = np.zeros((
            bsize,
            1
        ))
    
    for in_idx in range(bsize):
        
        img_path = urls[in_idx]
        X_train[in_idx,:,:,:] = processImg(img_path)
        Y_train[in_idx, 0] = getLabel(img_path)


    Y = Y_train.astype('int')
    Y = np_utils.to_categorical(Y, 10)
    X = X_train.astype('float32')
    X /= 255
    # return (X,[Y,Y,Y])
    return (X,Y)


def numSamples(bsize):
    return nTrain

def gen_batch(bsize = 64, isTest = False):

    while True:
        for in_idx in range(0,nTrain,bsize):

            urlBatch = train_data[ in_idx: min(tot, in_idx+bsize)]
            batch = processBatchUrls(urlBatch)
            yield batch
            

def get_test():
    return processBatchUrls(test_data)


def get_mean():
    mean = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
    for in_idx in range(tot):
        img_path = train_data[in_idx]

        img = cv2.imread(img_path, 0)
        img = transform_img(img, img_width=IMAGE_HEIGHT, img_height=IMAGE_WIDTH)
        mean += img[:,:,0]
        print in_idx

    return mean/tot

