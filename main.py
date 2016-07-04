from config import config
from soundProcessing import batchgen_and_numsamples

bge, numSamples = batchgen_and_numsamples()
print numSamples

print bge.next()[0].shape
