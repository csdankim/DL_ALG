"""
Dongkyu Kim
"""


from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
import math
import matplotlib.pyplot as plt
%matplotlib inline


def load_data():
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar-2class-py2/cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar-2class-py2/cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data'].T                     # 3072(32x32x3) x 10000
    train_y = data[b'train_labels'].T                   # 1 x 10000, label 0 is an airplane, label 1 is a ship
    test_x = data[b'test_data'].T                       # 3072 x 2000
    test_y = data[b'test_labels'].T                     # 1 x 2000
    
    # normalization of train_x and test_x
    train_x = (train_x - train_x.mean()) / train_x.std()
    test_x = (test_x - test_x.mean()) / test_x.std()
    
    return train_x, train_y, test_x, test_y


# initialize parameters for a neural network with one hidden layer
def initialize_parameters(input_dims, hidden_units, output_dims=1):
    
    np.random.seed(1)
    
    W1 = np.random.randn(hidden_units, input_dims)*0.01
    b1 = np.zeros((hidden_units, 1))
    W2 = np.random.randn(output_dims, hidden_units)*0.01
    b2 = np.zeros((output_dims, 1))
    
    assert(W1.shape == (hidden_units, input_dims))
    assert(b1.shape == (hidden_units, 1))
    assert(W2.shape == (output_dims, hidden_units))
    assert(b2.shape == (output_dims, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


class LinearTransform(object):

    def forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        
        return Z, cache

    def backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db
    

class ReLU(object):

    def forward(self, Z):
        A = np.maximum(0,Z)

        assert(A.shape == Z.shape)

        cache = Z 
        
        return A, cache

    def backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.

        # When z <= 0, should set dz to 0 as well. 
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ
    
    

class SigmoidCrossEntropy(object):
    
    def sigmoid_forward(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z

        return A, cache
    
    def sigmoid_backward(self, dA, cache):
        Z = cache

        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)

        assert (dZ.shape == Z.shape)

        return dZ

    def compute_cost(self, A2, Y):
        m = Y.shape[1]
        
        cost = - np.sum(np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))) / m
        
        cost = np.squeeze(cost)      # To make sure cost's shape is what I expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())

        return cost
    
    def backward_cost(self, A2, Y):
        dA2 = - (np.divide(Y, A2) - np.divide(1-Y, 1-A2))
        
        return dA2


class MLP(object):
    
    def __init__(self):
        # initialize all classes
        self.linear = LinearTransform()
        self.relu = ReLU()
        self.sigmoid_cost = SigmoidCrossEntropy()
        
    def forward_model(self, X, Y, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        Z1, linear_cache1 = self.linear.forward(X, W1, b1)
        A1, activation_cache1 = self.relu.forward(Z1)
        cache1 = (linear_cache1, activation_cache1)
        Z2, linear_cache2 = self.linear.forward(A1, W2, b2)
        A2, activation_cache2 = self.sigmoid_cost.sigmoid_forward(Z2)
        cache2 = (linear_cache2, activation_cache2)
        cost = self.sigmoid_cost.compute_cost(A2, Y)
        
        return A1, cache1, A2, cache2, cost
    
    def backward_model(self, A1, cache1, A2, cache2, Y):
        linear_cache2, activation_cache2 = cache2
        dA2 = self.sigmoid_cost.backward_cost(A2, Y)
        dZ2 = self.sigmoid_cost.sigmoid_backward(dA2, activation_cache2)
        dA1, dW2, db2 = self.linear.backward(dZ2, linear_cache2)
        linear_cache1, activation_cache1 = cache1
        dZ1 = self.relu.backward(dA1, activation_cache1)
        dA0, dW1, db1 = self.linear.backward(dZ1, linear_cache1)
        
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return grads
    
    def evaluate(self, A2, y):
        Y_prediction = np.zeros((y.shape))
        Y_prediction = np.where(A2<=0.5, 0.0, 1.0)
        
        assert(Y_prediction.shape == y.shape)
        
        return np.sum((Y_prediction == y))

    
# update parameters with SGD + momentum
class Optimizer(object):
    
    def __init__(self, learning_rate, momentum=None):
        self.learning_rate = learning_rate
        self.mu = momentum
        if self.mu:
            self.D = dict()
    
    def update_parameters(self, parameters, grads):
        if self.mu:
            for key in parameters.keys():
                if key not in self.D:
                    self.D[key] = np.zeros((grads["d" + key].shape))
                else:
                    self.D[key] = self.mu * self.D[key] - self.learning_rate * grads["d" + key]
                parameters[key] += self.D[key]
        else:
            parameters["W1"] -= self.learning_rate * grads["dW1"]
            parameters["b1"] -= self.learning_rate * grads["db1"]
            parameters["W2"] -= self.learning_rate * grads["dW2"]
            parameters["b2"] -= self.learning_rate * grads["db2"]
        
        return parameters


def random_mini_batches(X, Y, mini_batch_size, seed):
    
    np.random.seed(seed)            
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, -(m - mini_batch_size * num_complete_minibatches): ]
        mini_batch_Y = shuffled_Y[:, -(m - mini_batch_size * num_complete_minibatches): ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model(train_x, train_y, test_x, test_y, mini_batch_size, learning_rate, hidden_units, num_epochs, momentum, seed):
    
    input_dims_train, num_examples_train = train_x.shape
    
    _, num_examples_test = test_x.shape
    
    costs_train = []
    accuracies_train = []
    
    costs_test = []
    accuracies_test = []
    
    parameters = initialize_parameters(input_dims_train, hidden_units, output_dims=1)
    nn = MLP()
    optimizer = Optimizer(learning_rate, momentum)
    
    for epoch in range(num_epochs):
        seed = seed + 1
        minibatches_train = random_mini_batches(train_x, train_y, mini_batch_size, seed)
        cost_total_train = 0
        correct_total_train = 0
        
        ## train
        for i, minibatch_train in enumerate(minibatches_train):

            (minibatch_X, minibatch_Y) = minibatch_train

            A1, cache1, A2, cache2, cost = nn.forward_model(minibatch_X, minibatch_Y, parameters)

            cost_total_train += cost

            correct = nn.evaluate(A2, minibatch_Y)
            correct_total_train += correct

            grads = nn.backward_model(A1, cache1, A2, cache2, minibatch_Y)

            parameters = optimizer.update_parameters(parameters, grads)

            print('\rTrain[Epoch {}  mini-batch {}  Cost = {:.3f}]'.format(epoch, i, cost_total_train), end='')
            sys.stdout.flush()
        
        cost_avg_train = cost_total_train / num_examples_train
        costs_train.append(cost_avg_train)

        accuracy_train = correct_total_train / num_examples_train
        accuracies_train.append(accuracy_train)

        print()
        print('Train[Avg.cost: {}   Accuracy: {:.2f}%]'.format(cost_avg_train, 100. * accuracy_train))
        
        ## test
        _, _, A2_test, _, cost_test = nn.forward_model(test_x, test_y, parameters)
        
        costs_test.append(cost_test)
        
        correct_test = nn.evaluate(A2_test, test_y)
        accuracy_test = correct_test / num_examples_test
        accuracies_test.append(accuracy_test)
        
        print('Test[Cost: {}   Accuracy: {:.2f}%]'.format(cost_test, 100. * accuracy_test))
        print()
        
    results = {"costs_train": costs_train, 
               "accuracies_train": accuracies_train,
               "costs_test" : costs_test,
               "accuracies_test": accuracies_test}    

    return results


if __name__ == '__main__':
    
    train_x, train_y, test_x, test_y = load_data()
    
    # default hyper_parameters
    mini_batch_size = 64
    learning_rate = 0.001
    hidden_units = 100
    seed = 10
    num_epochs = 100
    momentum = 0.9

    
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_x, train_y, test_x, test_y, mini_batch_size, i, hidden_units, num_epochs, momentum, seed)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["accuracies_test"]), label= str(i))

    plt.ylabel('Test Accuracy')
    plt.xlabel('Iterations (epochs)')
    plt.title("Test Accuracy per Learning Rate")

    legend = plt.legend(loc='center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.savefig('learning_rate.png')
    plt.show()
    print ('\n' + "-------------------------------------------------------" + '\n')

    mini_batch_sizes = [64, 128, 356]
    models = {}
    for j in mini_batch_sizes:
        print ("mini batch size is: " + str(j))
        models[str(j)] = model(train_x, train_y, test_x, test_y, j, learning_rate, hidden_units, num_epochs, momentum, seed)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for j in mini_batch_sizes:
        plt.plot(np.squeeze(models[str(j)]["accuracies_test"]), label= str(j))

    plt.ylabel('Test Accuracy')
    plt.xlabel('Iterations (epochs)')
    plt.title("Test Accuracy per Mini_batch_size")

    legend = plt.legend(loc='center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.savefig('batch_size.png')
    plt.show()
    print ('\n' + "-------------------------------------------------------" + '\n')

    hidden_units_list = [10, 100, 1000]
    models = {}
    for k in hidden_units_list:
        print ("hidden unit is: " + str(k))
        models[str(k)] = model(train_x, train_y, test_x, test_y, mini_batch_size, learning_rate, k, num_epochs, momentum, seed)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for k in hidden_units_list:
        plt.plot(np.squeeze(models[str(k)]["accuracies_test"]), label= str(k))

    plt.ylabel('Test Accuracy')
    plt.xlabel('Iterations (epochs)')
    plt.title("Test Accuracy per hidden_unit")

    legend = plt.legend(loc='center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.savefig('hidden_units.png')
    plt.show()
    print ('\n' + "-------------------------------------------------------" + '\n')
    
    
    # the best hyper_parameters
    mini_batch_size = 64
    learning_rate = 0.0008
    hidden_units = 2000
    seed = 10
    num_epochs = 100
    momentum = 0.9
    
    best_results = model(train_x, train_y, test_x, test_y, mini_batch_size, learning_rate, hidden_units, num_epochs, momentum, seed)