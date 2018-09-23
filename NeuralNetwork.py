import numpy as np

class NeuralNetwork(object):
    def __init__(self, ils, hls, ols, nhl):

        #Parameters
        self.ils = ils #input layer size
        self.hls = hls #hidden layer size
        self.nhl = nhl #number of hidden layers
        self.ols = ols #output layer size

        self.batchSize = 0
        self.error = 0

        self.displayInitialization()
        self.initializeWeights()

    def displayInitialization(self):
        print("*******************************")
        print("* INITIALIZING NEURAL NETWORK *")
        print("*******************************")
        print("Input layer size:", self.ils)
        print("Hidden layer size:", self.hls)
        print("Output layer size:", self.ols)
        print("Number of hidden layers:", self.nhl)

    def initializeWeights(self):
        #Initialize input layer weights
        self.WI = np.random.randn(self.ils+1,self.hls)*0.01

        #Initialize hidden layer weights
        self.W = np.zeros((self.nhl-1,self.hls+1,self.hls))
        for i in range(len(self.W)):
            self.W[i] = np.random.randn(self.hls+1,self.hls)*0.01
            
        #Initialize output layer weights
        self.WO = np.random.randn(self.hls+1,self.ols)*0.01
    
    #Sigmoid activation function
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def forward(self, X, y=[0,0,0,0,0,0,0,0,0,0]):
        #Initialize data structures to store output from hidden layer (B)
        #and output layer (BO) neurons. Also, determine the batch size
        #based on outer dimension of input matrix.
        self.batchSize = len(X)
        self.B = np.zeros((self.nhl,self.batchSize,self.hls+1))
        self.BO = np.zeros((self.batchSize,self.ols))

        #Propagate input matrix (or vector) X through the network
        self.B[0] = np.c_[np.ones(self.batchSize), self.sigmoid(np.dot(X, self.WI))]
        for i in range(self.nhl-1):
            self.B[i+1] = np.c_[np.ones(self.batchSize), self.sigmoid(np.dot(self.B[i], self.W[i]))]
        self.BO = self.sigmoid(np.dot(self.B[len(self.B)-1], self.WO))

        #Use least squares error formula
        self.error = (.5)*np.sum(np.square(y-self.BO))
        
        return self.pickMax()
        
    def backprop(self, X, y, LR=0.02):
        #Initialize data structures for hidden layer (D)
        #and output layer (DO) deltas
        self.D = np.zeros((self.nhl, self.batchSize, self.hls))
        self.DO = np.zeros((self.batchSize,self.ols))
        
        #Calculate deltas for output layer neurons
        self.DO = np.multiply((self.BO - y), np.multiply(self.BO, np.subtract(1, self.BO)))
        
        #Calculate deltas for hidden layer neurons
        for i in range(self.nhl-1,-1,-1):
            if i == self.nhl-1:
                a = np.dot(self.DO, np.transpose(self.WO[1:,:]))
                b = np.multiply(self.B[i], np.subtract(1, self.B[i]))[:,1:]
                self.D[i] = np.multiply(a,b)
            else:
                a = np.dot(self.D[i+1], np.transpose(self.W[i][1:,:]))
                b = np.multiply(self.B[i], np.subtract(1, self.B[i]))[:,1:]
                self.D[i] = np.multiply(a,b)
        
        #Apply weight adjustments
        self.WI -= LR*np.dot(np.transpose(X), self.D[0])
        for i in range(self.nhl-1):
            self.W[i] -= LR*np.dot(np.transpose(self.B[i]), self.D[i+1])
        self.WO -= LR*np.dot(np.transpose(self.B[self.nhl-1]), self.DO)
    
    def pickMax(self):
        maxVal = 0
        maxIndex = 0
        for x in range(len(self.BO[0])):
            if self.BO[0][x] > maxVal:
                maxVal = self.BO[0][x]
                maxIndex = x
        return maxIndex
    
    def saveBrain(self, tag = ""):
        """
        Save the network's weights which constitute its brain. Pass in
        a number to uniquely identify this brain from others.
        """
        try:
            np.savez("brain"+str(tag), WI=self.WI, W=self.W, WO=self.WO)
            print("Successfully saved", "brain"+str(tag)+".npz")
        except:
            print("Failed to save", "brain"+str(tag)+".npz")
            
    def loadBrain(self, tag = ""):
        """
        Load a formerly saved set of network weights.
        """
        try:
            weights = np.load("brain"+str(tag)+".npz")
            self.WI = weights["WI"]
            self.W  = weights["W"]
            self.WO = weights["WO"]
            print("Successfully loaded", "brain"+str(tag)+".npz")
        except:
            print("Failed to load", "brain"+str(tag)+".npz")