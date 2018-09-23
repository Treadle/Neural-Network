import pickle
import gzip
import math
import numpy as np
from scipy import ndimage
from random import randint

class Data(object):
  def __init__(self):
    print("INITIALIZING DATA")
    print("Importing training data...")
    self.training = self.importData('data_training')
    print("Importing testing data...")
    self.testing = self.importData('data_testing')
    print("Normalizing data")
    self.normalizeData()
    
    print("Converting target data to one-hot encoding...")
    self.yTraining = []
    for target in self.training[1]:
      self.yTraining.append(self.convertToOneHot(target))

    self.yTesting = []
    for target in self.testing[1]:
      self.yTesting.append(self.convertToOneHot(target))
    print("DATA INITIALIZATION COMPLETE\n")

  def importData(self, filename):
    infile = gzip.open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

  def convertToOneHot(self, integer):
    vec = np.zeros(10)
    vec[integer] = 1
    return vec

  def normalizeData(self):
    if not self.loadNormalizedData():
      for f in range(len(self.training[0][0])):    #for every feature
        tmp = []
        for i in range(len(self.training[0])):   #for every vector
          tmp.append(self.training[0][i][f])

        mean = np.mean(tmp)
        std = np.std(tmp)
        for k in range(len(self.training[0])):
          if std > 0 and not math.isnan((self.training[0][k][f] - mean) / std) and np.isfinite((self.training[0][k][f] - mean) / std):
            self.training[0][k][f] = (self.training[0][k][f] - mean) / std
        for k in range(len(self.testing[0])):
          if std > 0 and not math.isnan((self.testing[0][k][f] - mean) / std) and np.isfinite((self.testing[0][k][f] - mean) / std):
            self.testing[0][k][f] = (self.testing[0][k][f] - mean) / std

      self.saveNormalizedData()

  def saveNormalizedData(self):
    try:
      np.savez("normalizedData", training=self.training[0], testing=self.testing[0])
      print("Successfully saved")
    except:
      print("Failed to save")

  def loadNormalizedData(self):
    try:
      normalizedData = np.load("normalizedData.npz")
      self.training[0] = normalizedData["training"]
      self.testing[0]  = normalizedData["testing"]
      print("Successfully loaded normalizedData.npz")
      return True
    except Exception as exception:
      print(str(exception))
      print("Failed to load normalizedData.npz")
      return False

  def generateRandomBatch(self, batchSize):
    X = np.zeros((batchSize, 785))
    y = np.zeros((batchSize, 10))
    for i in range(batchSize):
      randIndex = randint(0, 59999)
      X[i] = np.insert(self.training[0][randIndex],0,1)
      y[i] = self.yTraining[randIndex]
    return X, y