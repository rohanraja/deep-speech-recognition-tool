from module import Module
import scipy.io.wavfile as w
import numpy as np

# offsets = [0,0.5, -0.5, 0.25, -0.25]
offsets = [0]
volumes = [1,2]
bgFile = "bgNoise.wav"
bgsamp, bgdata = w.read(bgFile)
bgdata = bgdata[1::2]

def overlay(vdata, bgdata, offset=0 , volumeFactor=1):
    """
        Overlay audio having same sampling rate

        offset -> can be any ratio like (0, 0.5, -0.5)
    """

    bgN = bgdata.shape[0]
    vN = vdata.shape[0]
    stPos = (bgN - vN) / 2
    stPos += int(offset*stPos)

    overLayed = bgdata.copy()
    overLayed[stPos:stPos+vN] += volumeFactor*vdata

    return overLayed

def translations(vdata, samp):
    # return [vdata], samp
    results = []
    for off in offsets:
        for v in volumes:
            overlayed = overlay(vdata, bgdata, off, v)
            results.append(overlayed)

    for v in volumes:
        results.append(vdata*v)

    return results, samp

class WavVariations(Module):
    
    def __init__(self):

        self.allMethods = {
                "translations" : translations,
                "default": translations
        }
