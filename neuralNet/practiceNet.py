import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def neural_net(training_inputs, training_weights, hidden_weights, training_outputs, max_num_epochs,
learning_rate):

    for epoch in range(max_num_epochs):
        #feed forward
        hidden_layer = []

        for weight in training_weights:
            hidden_layer.append(sigmoid(np.dot(training_inputs, weight)))

        hidden_layer.append(1) #b1
        y = sigmoid(np.dot(hidden_layer, hidden_weights))

        #t = 1 #ohh t would be the correct answer
        total_network_error = 1/2*(np.square(training_outputs - y))

        #back propogation
        error_of_y = y*(1-y)*(training_outputs-y)

        error_array = []
        for index, element in enumerate(hidden_layer):
            error_array.append(element*(1-element)*(hidden_weights[index] * error_of_y))


        #learn
        #part 1 the hidden layer --> y
        for index, weight in enumerate(hidden_weights):
            hidden_weights[index] = hidden_weights[index] + learning_rate * error_of_y * hidden_layer[index]
        
        #part 2 the inputs --> hidden layer
        for j, weight in enumerate(training_weights):
            for k, variable in enumerate(weight):
                training_weights[j][k] = training_weights[j][k] + learning_rate * error_array[j] * training_inputs[k]

if __name__ == "__main__":

    file = open("practice.txt", "r")
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

    #training_weights = [[1, .5, 1], [-1, 2, 1]]
    #hidden_weights = [1.5, -1, 1]
    training_outputs = 1

    max_num_epochs = 1
    learning_rate = .5

    neural_net(training_inputs, training_weights, hidden_weights, training_outputs, max_num_epochs,
    learning_rate)