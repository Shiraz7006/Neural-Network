
# importing some useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn.preprocressing import StandardScaler
from sklearn import metrics
#from sklearn.metrics import confusion_matrix
import itertools
import pickle

# loads object from file
def unpickle(fileName):
    """
    takes filename as argument, returns dict object from pickle
    """
    import pickle
    with open(fileName, 'rb') as fileObj: 
        dict = pickle.load(fileObj, encoding='bytes')
    return dict

# defining a class for our 2-layer neuralnet
class neuralNet:
    """
    A class for implemeting a 2 layer neural network
    """
    def __init__(self, x, y):
        """
        CONSTRUCTOR FUNCTION FOR NEURAL NETWORK
        ---------------------------------------
        X = input
        Y = output (ideal)
        Yd = output (predicted)
        size = dimensions [input units, 1-L units, 2-L(out) units]
        lays = number of layers
        dwnb = dict to hold weights and bias
        temp = a temp variable for calculations
        lrte = learning rate
        nsmp = number of samples
        loss = stores Y-Yd every x iterations
        """
        self.X = x
        self.Y = y
        
        self.lays = 2 # 2-layerd
        self.size = [3072, 3072, 1]
        # self.dwnb = {'W1': [], 'B1': [], 'W2': [], 'B2': []}
        self.dwnb = {}
        self.temp = {}

        self.loss = []
        self.lrte = 0.002
        self.nsmp = (self.Y.shape[0])

        self.Yd = np.zeros((1, self.nsmp))

        return
    
    def randomInitializer(self):
        """
        Initializes the weights and biases with random values.
        """
        np.random.seed(1)
        self.dwnb['W1'] = (np.random.randn(self.size[1], self.size[0]) / np.sqrt(self.size[0]))
        self.dwnb['B1'] = (np.zeros((self.size[1], 1))) #column vector of height= n(1L units)
        self.dwnb['W2'] = (np.random.randn(self.size[2], self.size[1]) / np.sqrt(self.size[1]))
        self.dwnb['B2'] = (np.zeros((self.size[2], 1)))
        return
    
    def calcSigmoid(self, z):
        """
        calculates and returns sigmoid function for a particular input
        """
        return (1/(1 + np.exp(-z)))

    def calcNetLoss(self, yP):
        """
        calculate and return calculated loss
        """
        lossT = (1./self.nsmp) * ((0 - np.dot(self.Y, np.log(yP).T)) - np.dot(1-self.Y, np.log(1-yP).T))
        return lossT

    def calcDerSigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        dZ = s * (1-s)
        return dZ

    def passForward(self):
        """
        """
        z1 = (np.dot(self.dwnb['W1'], (self.X)) + self.dwnb['B1'])
        a1 = self.calcSigmoid(z1)
        self.temp['Z1'] = z1
        self.temp['A1'] = a1

        z2 = (np.dot(self.dwnb['W2'], (self.X)) + self.dwnb['B2'])
        a2 = self.calcSigmoid(z2)
        self.temp['Z2'] = z2
        self.temp['A2'] = a2

        self.Yd = a2
        loss = self.calcNetLoss(a2)
        return self.Yd, loss

    def passBackward(self):
        """
        """
        dlossYd = -(np.divide(self.Y, self.Yd) - np.divide(1-self.Y, 1-self.Yd))

        dlossZ2 = dlossYd * self.calcDerSigmoid(self.temp['Z2'])
        dlossA1 = np.dot(np.transpose(self.dwnb['W2']), dlossZ2)
        dlossW2 = ((1./(self.temp['A1'].shape[1])) * np.dot(dlossZ2, np.transpose(self.temp['A1'])))
        dlossB2 = ((1./(self.temp['A1'].shape[1])) * np.dot(dlossZ2, np.ones([dlossZ2.shape[1], 1])))

        dlossZ1 = dlossA1 * self.calcDerSigmoid(self.temp['Z1'])
        dlossA0 = np.dot(np.transpose(self.dwnb['W1']), dlossZ1)
        dlossW1 = ((1./(self.X.shape[1])) * np.dot(dlossZ1, np.transpose(self.X)))
        dlossB1 = ((1./(self.X.shape[1])) * np.dot(dlossZ1, np.ones([dlossZ1.shape[1], 1])))

        self.dwnb['W1'] = self.dwnb['W1'] - self.lrte * dlossW1
        self.dwnb['B1'] = self.dwnb['B1'] - self.lrte * dlossB1
        self.dwnb['W2'] = self.dwnb['W2'] - self.lrte * dlossW2
        self.dwnb['B2'] = self.dwnb['B2'] - self.lrte * dlossB2
        
        return
    
    def gradientDescent(self, n):
        """
        """
        np.random.seed(1)
        self.randomInitializer()
        for i in range(n):
            yP, loss = self.passForward()
            self.passBackward()
            if(i % 200):
                print("Cost after iteration %i: %f" %(i, loss))
                self.loss.append(loss)
        return

#--- END of CLASS neuralNet

dataA = unpickle('data_batch_2')

# print(' >> KEYS:')
# for i in dataA:
#     print(i)

# KEYS: b'batch_label', b'labels', b'data', b'filenames'

#y = np.transpose(dataA[b'labels'])
y = (np.array(dataA[b'labels']))
xtemp = dataA[b'data']

scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(xtemp)
print(xtemp)
print(x)
print(y.shape)
print(x.shape)

testClass = neuralNet(np.transpose(x), np.transpose(y))
testClass.gradientDescent(5)

# print(x[0].shape)