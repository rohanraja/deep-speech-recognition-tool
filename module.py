class Module():
    
    def get_method(self, methodName):
        return self.allMethods[methodName]

from wavToimg import WavToImg
from wavVariations import WavVariations
modules = {}
modules["wavToImage"] = WavToImg
modules["wavVariations"] = WavVariations





from config import config

def getModule(modName):

    modClass = modules[modName]()
    return modClass.get_method(config.get(modName,"default"))
