import sys
import json
import os

def loadConfig():

    cFile = os.path.join(sys.argv[1], "config.json")
    config = json.load(open(cFile, 'r'))

    return config


config = loadConfig()

