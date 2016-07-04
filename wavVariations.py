from module import Module

def translations(data, samp):
    return [data], samp

class WavVariations(Module):
    
    def __init__(self):

        self.allMethods = {
                "translations" : translations,
                "default": translations
        }



