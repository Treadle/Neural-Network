import numpy as np

class NeuralNetwork(object):
  def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, number_of_hidden_layers):
    self.input_layer_size = input_layer_size
    self.hidden_layer_size = hidden_layer_size
    self.number_of_hidden_layers = number_of_hidden_layers
    self.output_layer_size = output_layer_size
    self.number_of_synapses = input_layer_size * hidden_layer_size \
      + hidden_layer_size * (number_of_hidden_layers - 1) \
      + hidden_layer_size * output_layer_size

    self.displayInitialization()
    self.initializeWeights()

  def displayInitialization(self):
    print("*******************************")
    print("* INITIALIZING NEURAL NETWORK *")
    print("*******************************")
    print("Input layer size:", self.input_layer_size)
    print("Hidden layer size:", self.hidden_layer_size)
    print("Output layer size:", self.output_layer_size)
    print("Number of hidden layers:", self.number_of_hidden_layers)
    print("Total number of synapses:", self.number_of_synapses)
    print()

  def initializeWeights(self):
    self.input_layer_weights = np.random.randn(
      self.input_layer_size + 1,
      self.hidden_layer_size
    ) * 0.01

    self.hidden_layer_weights = np.zeros((
      self.number_of_hidden_layers - 1,
      self.hidden_layer_size + 1,
      self.hidden_layer_size
    ))
    for layer in self.hidden_layer_weights:
      layer = np.random.randn(
        self.hidden_layer_size + 1,
        self.hidden_layer_size
      ) * 0.01

    self.output_layer_weights = np.random.randn(
      self.hidden_layer_size + 1,
      self.output_layer_size
    ) * 0.01

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def forward(self, X, y=[0,0,0,0,0,0,0,0,0,0]):
    # Initialize hidden and output layer outputs
    self.batch_size = len(X)

    self.hidden_layer_outputs = np.zeros((
      self.number_of_hidden_layers,
      self.batch_size,
      self.hidden_layer_size + 1
    ))
    self.output_layer_outputs = np.zeros((
      self.batch_size,
      self.output_layer_size
    ))

    # Forward propagate
    self.hidden_layer_outputs[0] = np.c_[
      np.ones(self.batch_size),
      self.sigmoid(
        np.dot(
          X,
          self.input_layer_weights
        )
      )
    ]

    for layer in range(self.number_of_hidden_layers - 1):
      self.hidden_layer_outputs[layer + 1] = np.c_[
        np.ones(self.batch_size),
        self.sigmoid(
          np.dot(
            self.hidden_layer_outputs[layer],
            self.hidden_layer_weights[layer]
          )
        )
      ]

    self.output_layer_outputs = self.sigmoid(
      np.dot(
        self.hidden_layer_outputs[-1],
        self.output_layer_weights
      )
    )

    # Least squares error
    self.error = (0.5) * np.sum(np.square(y - self.output_layer_outputs))

    return self.pickMax()

  def backprop(self, X, y, learning_rate = 0.02):
    # Initialize deltas
    self.hidden_layer_deltas = np.zeros((
      self.number_of_hidden_layers,
      self.batch_size,
      self.hidden_layer_size
    ))
    self.output_layer_deltas = np.zeros((
      self.batch_size,
      self.output_layer_size
    ))

    # Calculate deltas
    self.output_layer_deltas = np.multiply(
      self.output_layer_outputs - y,
      np.multiply(
        self.output_layer_outputs,
        np.subtract(1, self.output_layer_outputs)
      )
    )

    for layer in range(self.number_of_hidden_layers-1, -1, -1):
      self.hidden_layer_deltas[layer] = np.multiply(
        np.dot(
          self.output_layer_deltas,
          np.transpose(self.output_layer_weights[1:, :])
        )
        if layer == self.number_of_hidden_layers - 1 else 
        np.dot(
          self.hidden_layer_deltas[layer + 1],
          np.transpose(self.hidden_layer_weights[layer][1:, :])
        ),
        np.multiply(
          self.hidden_layer_outputs[layer],
          np.subtract(1, self.hidden_layer_outputs[layer])
        )[:, 1:]
      )

    # Apply updates
    self.input_layer_weights -= learning_rate * np.dot(
      np.transpose(X),
      self.hidden_layer_deltas[0]
    )

    for layer in range(self.number_of_hidden_layers - 1):
      self.hidden_layer_weights[layer] -= learning_rate * np.dot(
        np.transpose(self.hidden_layer_outputs[layer]),
        self.hidden_layer_deltas[layer + 1]
      )

    self.output_layer_weights -= learning_rate * np.dot(
      np.transpose(self.hidden_layer_outputs[self.number_of_hidden_layers - 1]),
      self.output_layer_deltas
    )

  def pickMax(self):
    return np.fromiter(
      map(lambda output : output.argmax(), self.output_layer_outputs),
      dtype=np.int
    )
  
  def saveBrain(self, tag = ""):
    try:
      np.savez(
        "brain"+str(tag),
        WI = self.input_layer_weights,
        W = self.hidden_layer_weights,
        WO = self.output_layer_weights)
      print("Successfully saved brain" + str(tag) + ".npz")
    except:
      print("Failed to save brain" + str(tag) + ".npz")
          
  def loadBrain(self, tag = ""):
    try:
      weights = np.load("brain" + str(tag) + ".npz")
      self.input_layer_weights = weights["WI"]
      self.hidden_layer_weights  = weights["W"]
      self.output_layer_weights = weights["WO"]
      print("Successfully loaded brain" + str(tag) + ".npz")
    except:
      print("Failed to load brain" + str(tag) + ".npz")