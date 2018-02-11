import pickle
import gzip
import math
import numpy as np
from scipy import ndimage
from random import randint

class Data(object):
    def __init__(self):
        print("DATA INITIALIZATION BEGINNING")
        print("Importing training data...")
        self.training = self.importData('data_training')
        print("Importing testing data...")
        self.testing = self.importData('data_testing')
        print("Normalizing data")
        self.normalizeData(self.training, self.testing)
        
        print("Converting target data to one-hot encoding...")
        self.yTraining = []
        for target in self.training[1]:
            self.yTraining.append(self.convertToOneHot(target))

        self.yTesting = []
        for target in self.testing[1]:
            self.yTesting.append(self.convertToOneHot(target))
        print("DATA INITIALIZATION COMPLETE")

    def importData(self, filename):
        infile = gzip.open(filename, 'rb')
        data = pickle.load(infile)
        infile.close()
        return data

    def convertToOneHot(self, integer):
        vec = np.zeros(10)
        vec[integer] = 1
        return vec

    def normalizeData(self, training, testing):
        for f in range(len(training[0][0])):    #for every feature
            tmp = []
            for i in range(len(training[0])):   #for every vector
                tmp.append(training[0][i][f])

            mean = np.mean(tmp)
            std = np.std(tmp)
            for k in range(len(training[0])):
                if std > 0 and not math.isnan((training[0][k][f] - mean) / std) and np.isfinite((training[0][k][f] - mean) / std):
                    training[0][k][f] = (training[0][k][f] - mean) / std
            for k in range(len(testing[0])):
                if std > 0 and not math.isnan((testing[0][k][f] - mean) / std) and np.isfinite((testing[0][k][f] - mean) / std):
                    testing[0][k][f] = (testing[0][k][f] - mean) / std
        return training, testing

    def generateRandomBatch(self, batchSize):
        X = np.zeros((batchSize, 785))
        y = np.zeros((batchSize, 10))
        for i in range(batchSize):
            randIndex = randint(0, 59999)
            X[i] = np.insert(self.training[0][randIndex],0,1)
            y[i] = self.yTraining[randIndex]
        return X, y