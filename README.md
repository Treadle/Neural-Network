# neural-network
The project is currently tailored to the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten digits. Logic is separated into three classes, ```Data.py```, ```NeuralNetwork.py```, and ```Trainer.py```. The ```Main.py``` file conducts the interaction between the aforementioned three classes. This project is ongoing. Next steps include
* writing a better ```README.md``` and
* making the project generalize to work with other datasets.

Sample output: 
```
*******************************
* INITIALIZING NEURAL NETWORK *
*******************************
Input layer size: 784
Hidden layer size: 128
Output layer size: 10
Number of hidden layers: 2
***************************
* TRAINING NEURAL NETWORK *
***************************
Epochs: 12
Iterations per epoch: 1875
Batch size: 32
Learning rate: 0.02

epoch    training        testing
0        20.2%           20.15%
1        29.13%          29.51%
2        85.52%          85.36%
3        91.8%           91.58%
4        93.79%          93.28%
5        94.91%          94.17%
6        95.4%           94.3%
7        96.16%          94.97%
8        96.58%          95.48%
9        96.94%          95.6%
10       97.39%          95.85%
11       97.57%          95.88%
```