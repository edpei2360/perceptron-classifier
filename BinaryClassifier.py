import numpy as np
import matplotlib.pyplot as plt

class BinaryPerceptron():

    def __init__(self, featureVectorLength):
        self.featureVectorLength = featureVectorLength
        self.weight = np.zeros(featureVectorLength)
        self.beta = 0
        self.trainingMistakes = []
        self.testMistakes = 0
        
    # Purpose: Trains a binary perceptron based on a set of data 
    # Parameters:
    #   sampleData - array of feature vectors
    #   sampleDataLabels - array of containing labels for feature vectors in sampleData (must be binary labels of -1,1)
    #   max_pass - number of training iterations for the perceptron
    def TrainBinaryPerceptron(self, sampleData, sampleDataLabels, max_pass):
        for passNumber in range(max_pass):
            self.trainingMistakes.append(0) 
            for i in range(len(sampleData)):
                if ((sampleDataLabels[i] * (np.dot(sampleData[i], self.weight) + self.beta))[0] <= 0):
                    # punish perceptron when it is indecisive or wrong
                    self.weight = np.add(self.weight, np.multiply(sampleDataLabels[i], sampleData[i]))
                    self.beta += sampleDataLabels[i]
                    self.trainingMistakes[passNumber] += 1


    # Purpose: Tests a binary perceptron based on its weight vectory and beta value 
    # Parameters:
    #   sampleData - array of feature vectors
    #   sampleDataLabels - array of containing labels for feature vectors in sampleData (must be binary labels of -1,1)
    def TestBinaryPerceptron(self, sampleData, sampleDataLabels):
        self.testMistakes = 0

        # run through given data points
        for dataIter in range(len(sampleData)):

            prediction = np.dot(sampleData[dataIter], np.array(self.weight)) + self.beta

            if (prediction < 0):
                prediction = -1
            elif (prediction > 0):
                prediction = 1
            else:
                prediction = 0  # if we predict 0 algorithm is indecicive, so mark it as wrong

            # check if perceptrons were able to predict the correct label
            if ( prediction != sampleDataLabels[dataIter]):
                self.testMistakes += 1

    # Purpose: show a graph plotting the mistakes made by the perceptron at each training iteration
    def PlotMistakes(self):
        plt.plot(range(len(self.trainingMistakes)), self.trainingMistakes)
        plt.xlabel("training iteration")
        plt.ylabel("errors made")
        plt.show()
        