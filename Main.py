from BinaryClassifier import BinaryPerceptron
import DataLoader 
import numpy as np

def Main():

    rawSampleData = DataLoader.LoadRawFeatureVectors("data/spambase_X.csv")
    rawSampleDataLabels = DataLoader.LoadRawFeatureVectors("data/spambase_Y.csv")

    # preform data formatting under the assumption that each column
    # represents a new single data point
    featureVecCount = len(rawSampleData)

    sampleData = []
    sampleDataLabels = []
    
    for column in rawSampleData.columns:
        sampleData.append(np.array(rawSampleData[column]))

    # in our instance labels are all in one column so transpose to
    # put each label in its own column to match feature vector format
    rawSampleDataLabels = rawSampleDataLabels.transpose()

    for column in rawSampleDataLabels.columns:
        sampleDataLabels.append(np.array(rawSampleDataLabels[column]))

    classifier = BinaryPerceptron(featureVecCount)

    # train the perceptron
    classifier.TrainBinaryPerceptron(sampleData, sampleDataLabels, 500)

    # plot mistakes to see how effective train was
    classifier.PlotMistakes()

# call main function
Main()