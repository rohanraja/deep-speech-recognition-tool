from config import config
from model import getModel
import os
import cv2
import glob
import numpy as np
import scipy.io.wavfile as wv
from module import getModule
from soundProcessing import make_trainable

model = getModel()

def processwavFile(wavPath, addBg = False):

    sampRate, wData = wv.read(wavPath)

    if addBg:
        sampRate /= 2
        variationsWav, sampRate = getModule("wavVariations")(wData[1::2], sampRate)
        wData = variationsWav[0]
    img = getModule("wavToImage")(wData, sampRate)

    X, _ = make_trainable( (img,0)  ) 
    return np.array([X], dtype="float32")


def predict(wavPath):

    x = processwavFile(wavPath, True)
    pred = model.predict(x)
    print pred
    # return str(pred[1][0].argmax())
    return str(pred[0].argmax())
    

def updateWeight(wid = -1):

    wghts = glob.glob(os.path.join(config["projectDir"], "savedModel_*"))
    
    nums = map(lambda w: int(w.split("/")[-1].split("_")[1]) , wghts)
    if wid == -1:
        num = max(nums)
    else:
        num = wid

    print("Updating to weight #%d" % num)
    model.load_weights(os.path.join(config["projectDir"], "savedModel_%d"%num))

    return num

updateWeight()

# print predict("/home/rohan/code/caffe/caffe-speech-recognition/spoken_numbers4/4_Albert_180.wav")
