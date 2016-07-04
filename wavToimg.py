from module import Module
from features import mfcc
from features import logfbank
import numpy as np
import matplotlib.pyplot as plt
import cv2

def MFCC(data, samp):

    mfcc_feat = mfcc(data,samp)
    mMin = mfcc_feat.min()
    mMax = mfcc_feat.max()
    mfcc_feat -= mMin
    mfcc_feat *= 255/mfcc_feat.max()
    outImg = np.array(mfcc_feat, np.uint8)
    return outImg

def LogFBank(data, samp):

    mfcc_feat = logfbank(data,samp)
    mMin = mfcc_feat.min()
    mMax = mfcc_feat.max()
    mfcc_feat -= mMin
    mfcc_feat *= 255/mfcc_feat.max()
    outImg = np.array(mfcc_feat, np.uint8)
    return outImg

def LinearSpectrum(data, samp):

    import ipdb; ipdb.set_trace()
    fpath = "/tmp/fft.png"
    plt.clf()
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    frame1.spines['top'].set_visible(False)
    frame1.spines['bottom'].set_visible(False)
    frame1.spines['left'].set_visible(False)
    frame1.spines['right'].set_visible(False)
    S, freqs, bins, im = plt.specgram(data, NFFT=1024, Fs=samp, noverlap=512)

    plt.tight_layout()
    plt.axis('tight')
    
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
    return cv2.imread(fpath)

class WavToImg(Module):
    
    def __init__(self):

        self.allMethods = {
                "MFCC" : MFCC,
                "LinearSpectrum" : LinearSpectrum,
                "logFbank" : LogFBank,
                "default": MFCC
        }


