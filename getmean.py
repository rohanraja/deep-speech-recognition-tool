
from datagen import get_mean


import cv2

meanImg = get_mean()
cv2.imwrite("mean.png", meanImg)
