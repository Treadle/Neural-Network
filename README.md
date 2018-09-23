# neural-network
The project is currently tailored to the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten digits. Logic is separated into three classes, ```Data.py```, ```NeuralNetwork.py```, and ```Trainer.py```. The ```Main.py``` file conducts the interaction between the aforementioned three classes. This project is ongoing. Next steps include
* writing a better ```README.md``` and
* making the project generalize to work with other datasets.

Sample output: 
```
INITIALIZING DATA
Importing training data...
Importing testing data...
Normalizing data
Successfully loaded normalizedData.npz
Converting target data to one-hot encoding...
DATA INITIALIZATION COMPLETE

*******************************
* INITIALIZING NEURAL NETWORK *
*******************************
Input layer size: 784
Hidden layer size: 128
Output layer size: 10
Number of hidden layers: 2
Total number of synapses: 101760

***************************
* TRAINING NEURAL NETWORK *
***************************
Epochs: 12
Iterations per epoch: 937
Batch size: 64
Learning rate: 0.02

epoch    training        testing
0        15.23%          16.26%
1        74.31%          74.51%
2        89.61%          89.5%
3        92.39%          92.1%
4        94.06%          93.54%
5        95.15%          94.51%
6        95.83%          95.17%
7        96.41%          95.41%
8        96.85%          95.69%
9        97.29%          95.92%
10       97.56%          96.13%
11       97.88%          96.25%
```