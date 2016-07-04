from config import config

from soundProcessing import batchgen_and_numsamples

from model import getModel

from keras import callbacks


bgen, numSamps = batchgen_and_numsamples()

model = getModel()


class SaveEpoch(callbacks.Callback):
    def on_epoch_end(self, epch, thrd = {}):

        return
        model.save_weights("savedModel_%d"%epch, overwrite=True)


model.fit_generator(bgen,
        samples_per_epoch=numSamps,
        nb_epoch= int(config["epochs"]) ,
        max_q_size = 100,
        callbacks=[SaveEpoch()]
)
