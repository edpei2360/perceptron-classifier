import pandas as pd
import os

# NOTE: we can load data with these function under the assumption that
#       the data is in the same directory as our executing main

def LoadRawFeatureVectors(sampleDataFileLocation):
    sampleDataRaw = pd.read_csv(os.getcwd() + "/" + sampleDataFileLocation, header=None)
    return sampleDataRaw

def LoadRawLabels(sampleDataLabelsFileLocation):
    sampleDataLabelsRaw = pd.read_csv(os.getcwd() +  "/" + sampleDataLabelsFileLocation, header=None)
    return sampleDataLabelsRaw