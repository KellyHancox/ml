import numpy as np
import matplotlib.pyplot as plt
import re

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def neural_net(training_inputs, training_weights, hidden_weights, training_outputs, max_num_epochs,
learning_rate, bias, bias_weight, hidden_bias, hidden_bias_weight):

    for epoch in range(max_num_epochs):
        #feed forward
        hidden_layer = []

        for index, weight in enumerate(training_weights):
            for input in training_inputs:
                input = list(map(int, input))
                hidden_layer_node = np.dot(input, weight) + (bias * bias_weight[index])
                hidden_layer.append(sigmoid(hidden_layer_node))

        
        y = []
        for hidden_bias_index in hidden_bias_weight:
            print(hidden_bias_index)
            node = np.dot(hidden_layer, hidden_weights) + (hidden_bias * hidden_bias_index)
            #will become softmax
            y.append(sigmoid(node))

        #delta = (ideal[1] - actual[1])^2 + (ideal[2] -actual[2])^2 + ... + (ideal[n]-actual[n])^2  
        #mean square error = deta / n

        #t = 1 #ohh t would be the correct answer
        subtracted = np.array(training_outputs) - np.array(y)
        total_network_error = 1/2*(np.square(subtracted))
        #print(total_network_error)
        #returns as an array, with correct answer

        #back propogation
        #will become for loop with y as an array
        error_of_y = []
        for index, output in enumerate(y):
            error_of_y.append(output*(1-output)*(float(training_outputs[index])-output))
        print(error_of_y)
        #this pops out as: [array([0.03734276])]
        
        error_array = []
        for error_in_y in error_of_y:
            for index, element in enumerate(hidden_layer):

                error_array.append(element*(1-element)*(float(hidden_weights[index]) * float(error_in_y)))
                #change bias weight
        print(error_array)

        #learn
        #part 1 the hidden layer --> y
        for top_index, y_error in enumerate(error_of_y):
            for index, weight in enumerate(hidden_weights):
                hidden_weights[index] = hidden_weights[index] + learning_rate * y_error * hidden_layer[index]
                #change hidden bias weight
            hidden_bias_weight[top_index] = hidden_bias_weight[top_index] + learning_rate * y_error * hidden_bias
        
        #part 2 the inputs --> hidden layer
        for j, weight in enumerate(training_weights):
            input_index = 0
            for k, variable in enumerate(weight):
                training_weights[j][k] = float(training_weights[j][k]) + (learning_rate * error_array[j] * float(training_inputs[input_index][k]))
            
            bias_weight[j] = bias_weight[j] + learning_rate * error_array[j] * bias

        print(training_weights)
        print(hidden_weights)
        print(bias_weight)
        print(hidden_bias_weight)

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
    
    for input in training_inputs:
        input = list(map(int, input))
    training_outputs = list(map(int, training_outputs))

    training_weights = [[1, .5], [-1, 2]]
    hidden_weights = [1.5, -1]

    bias = 1
    #len is len(training_weights)
    bias_weight = [1, 1]
    hidden_bias = 1
    #len is len(final_outputs)
    hidden_bias_weight = [1]

    max_num_epochs = 1
    learning_rate = .5

    neural_net(training_inputs, training_weights, hidden_weights, training_outputs, max_num_epochs,
    learning_rate, bias, bias_weight, hidden_bias, hidden_bias_weight)