from soundProcessing import batch_generator, num_samples
import cv2

import os

spoDir = "/home/rohan/code/caffe/caffe-speech-recognition/spoken_numbers4"
# spoDir = "/home/rohan/code_mac/audiorecord/testset"

wavFiles = []

# fX = open("wavfiles.txt", 'w')
#
for f in os.listdir(spoDir):
    if f.endswith('.wav'):
        wavFiles.append(os.path.join(spoDir, f))



import random

random.shuffle(wavFiles)

bgen = batch_generator(32, wavFiles)




# imgs = processWav("/home/rohan/code/caffe/caffe-speech-recognition/spoken_numbers4/0_Agnes_100.wav")
# print imgs
