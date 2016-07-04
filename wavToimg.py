from module import Module
from features import mfcc
import numpy as np

def MFCC(data, samp):

    mfcc_feat = mfcc(data,samp)
    mMin = mfcc_feat.min()
    mMax = mfcc_feat.max()
    mfcc_feat -= mMin
    mfcc_feat *= 255/mfcc_feat.max()
    outImg = np.array(mfcc_feat, np.uint8)
    return outImg

class WavToImg(Module):
    
    def __init__(self):

        self.allMethods = {
                "MFCC" : MFCC,
                "default": MFCC
        }


