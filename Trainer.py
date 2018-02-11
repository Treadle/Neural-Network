import pickle
import gzip
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from random import randint
import time

class Trainer(object):
    def __init__(self, network, data):
        self.network = network
        self.data = data
        self.trainingPerformance = []
        self.testingPerformance = []
        
    def train(self, e = 10, bs = 32, lr=0.05):
        epochs = e
        batchSize = bs
        iters = int(60000 / batchSize)
        learningRate = lr
        
        print("***************************")
        print("* TRAINING NEURAL NETWORK *")
        print("***************************")
        print("Epochs:", epochs)
        print("Iterations per epoch:", iters)
        print("Batch size:", batchSize)
        print("Learning rate:", lr,"\n")
        print("epoch\t training\t testing")
        
        for i in range(epochs):
            for j in range(iters):
                X, y = self.data.generateRandomBatch(batchSize)
                self.network.forward(X, y)
                self.network.backprop(X, y, learningRate)
            trainingAccuracy, testingAccuracy = self.test()
            print(i,"\t ",trainingAccuracy,"%", "\t\t ", testingAccuracy, "%", sep="")

    def test(self):
        trainingAccuracy = self.accuracy(self.data.training[0], self.data.training[1])
        testingAccuracy = self.accuracy(self.data.testing[0], self.data.testing[1])
        self.trainingPerformance.append(trainingAccuracy)
        self.testingPerformance.append(testingAccuracy)
        return trainingAccuracy, testingAccuracy

    def accuracy(self, xData, yData):
        numCorrect = 0
        for j in range(len(xData)):
            X = np.expand_dims(np.insert(xData[j],0,1), axis=0)
            y = yData[j]
            if self.network.forward(X, y) == yData[j]:
                numCorrect += 1
        return round((numCorrect * 100) / len(xData),2)
