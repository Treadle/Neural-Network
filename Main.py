import Data as d
import Trainer as t
import NeuralNetwork as nn

data = d.Data()
network = nn.NeuralNetwork(784, 128, 10, 2)
trainer = t.Trainer(network, data)

trainer.train(12, 32, 0.02)