import sys
import json
import os

def loadConfig():

    cFile = os.path.join(sys.argv[1], "config.json")
    config = json.load(open(cFile, 'r'))
    config["projectDir"] = sys.argv[1]

    return config


config = loadConfig()

