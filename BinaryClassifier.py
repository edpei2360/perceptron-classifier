import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BinaryPerceptron:

    def __init__(self):
        self.weight = []
        self.beta = 0
        self.mistakes = 0        
        
    # Purpose: Trains a binary perceptron based on a set of data 
    # Parameters:
    #   sampleData - array of feature vectors
    #   sampleDataLabels - array of containing labels for feature vectors in sampleData
    def TrainBinaryPerceptron(self, sampleData, sampleDataLabels, max_pass):
        for passNumber in range(max_pass):
            self.mistakes.append(0) 
            for i in range(len(sampleData)):
                if (sampleDataLabels[i] * (np.dot(sampleData[i], self.weight) + self.beta) <= 0):
                    self.weight = np.add(self.weight, np.multiply(sampleDataLabels[i], sampleData[i]))  
                    self.beta += sampleDataLabels[i]
                    self.mistakes[passNumber] += 1
