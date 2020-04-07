#combo of https://www.youtube.com/watch?v=kft1AJ9WVDk and https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/

import numpy as np
import re

class NeuralNetwork():

    def __init__(self):
        np.random.seed(42)
        self.synaptic_weights = np.random.rand(64,1)
        self.bias = np.random.rand(1)
        self.lr = 0.05

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    #putting feed forward and back propogation together
    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):
            output = self.feed_forward(training_inputs)

            #backpropogation step 1: get difference of error
            error = output - training_outputs
            print(error.sum())

            #matrix multiplication
            delta = error * self.sigmoid_derivative(output)

            #print(training_inputs)
            training_inputs = training_inputs.astype(float)
            adjustments = self.lr * np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            #not sure if this is plus or minus...
            self.synaptic_weights -= adjustments

            #update bias
            for num in delta:
                self.bias -= self.lr * num
    
    def feed_forward(self, inputs):
        #using dot product again. our inputs are integers, our synaptic weights are floats
        #so we'll have to change inputs to floats

        #feed foward step 1 and 2 combined. 1 is the dot product, 2 is the sigmoid
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights) + self.bias)
        return output 


if __name__ == "__main__":
    neural_network = NeuralNetwork()
    
    print("random starting synaptic weights: ")
    print(neural_network.synaptic_weights)


    '''
    Strong,Warm,Warm,Sunny,Yes
    Weak,Warm,Warm,Sunny,No
    Strong,Warm,Warm,Cloudy,Yes
    Strong,Moderate,Warm,Rainy,Yes
    Strong,Cold,Cool,Rainy,No
    Weak,Cold,Cool,Rainy,No
    Weak,Cold,Cool,Sunny,No
    Strong,Moderate,Warm,Sunny,Yes
    Strong,Cold,Cool,Sunny,Yes
    Strong,Moderate,Cool,Rainy,No
    Weak,Moderate,Cool,Sunny,Yes
    Weak,Moderate,Warm,Sunny,Yes
    Strong,Warm,Cool,Sunny,Yes
    Weak,Moderate,Warm,Rainy,No

    0: strong, warm warm sunny
    1: weak, moderate, cool, Cloudy
    2: -, cold, -, rainy
    '''
    # training_inputs = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    # training_outputs = np.array([[1,0,0,1,1]]).T


    file = open("digitsTrain.txt", "r")
    contents = file.read()
    lines = re.split(r"\n", contents)

    training_outputs = []
    training_inputs = []

    for line in lines:
        split_line = re.split(r" ", line)
        training_outputs.append(split_line.pop())
        training_inputs.append(split_line)

    #print(training_inputs)
    training_inputs = np.array(training_inputs)
    
    training_outputs = np.array([training_outputs]).T
    training_outputs = training_outputs.astype(float)

    # #training iterations are a custom input
    neural_network.train(training_inputs, training_outputs, 1000)


    print("syn weights after training: ")
    print(neural_network.synaptic_weights)


    # A = str(input("Input 1: "))
    # B = str(input("Input 2: "))
    # C = str(input("Input 3: "))
    # D = str(input("Input 4: "))

    # print("new situation: input data = ", A, B, C, D)
    # print("output data: ")
    print(neural_network.feed_forward(np.array([0,0,12,10,0,0,0,0,0,0,14,16,16,14,0,0,0,0,13,16,15,10,1,0,0,0,11,16,16,7,0,0,0,0,0,4,7,16,7,0,0,0,0,0,4,16,9,0,0,0,5,4,12,16,4,0,0,0,9,16,16,10,0,0])))

    