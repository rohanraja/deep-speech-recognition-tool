from model import getModel
from datagen import transform_img, getLabel
import cv2
import glob
import numpy as np



model = getModel()

model.load_weights("savedModel_%d"%20)

def processImg(img_path):

    img = cv2.imread(img_path, 0)
    img = transform_img(img)
    x = [np.rollaxis(img, 2)]


    return np.array(x)


def predict(imgpath):

    x = processImg(imgpath)
    pred = model.predict(x)

    # return str(pred[1][0].argmax())
    return str(pred[0].argmax())
    

def updateWeight():

    wghts = glob.glob("savedModel_*")
    
    nums = map(lambda w: int(w.split("_")[1]) , wghts)
    num = max(nums)


    print("Updating to weight #%d" % num)
    model.load_weights("savedModel_%d"%num)

# updateWeight()
