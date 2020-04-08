import numpy as np
import matplotlib.pyplot as plt
import re

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    print(A)
    expA = np.exp(A)
    return expA/expA.sum()

def practice():
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
    hidden_bias_weights = [1]

    max_num_epochs = 1
    learning_rate = .5

    neural_net(training_inputs, training_weights, hidden_weights, training_outputs, max_num_epochs,
    learning_rate, bias, bias_weight, hidden_bias, hidden_bias_weights)

def neural_net(training_inputs, training_weights, hidden_weights, training_outputs, max_num_epochs,
learning_rate, bias, bias_weight, hidden_bias, hidden_bias_weights):

    for epoch in range(max_num_epochs):
        #feed forward

        hidden_layer = []

        #I want 48 nodes out of this. that are 48 x 3000 
        for index, weight in enumerate(training_weights):
            layer = []
            for input in training_inputs:

                hidden_layer_node = np.dot(input, weight) + (bias * bias_weight[index])
                #hidden_layer.append()
                layer.append(hidden_layer_node)
            layer_sigmoid = sigmoid(sum(layer))
            hidden_layer.append(layer_sigmoid)

        # print(len(hidden_layer))
        # #print(len(hidden_layer[0]))
        # print(len(hidden_weights))
        # print(len(hidden_weights[0]))
        y = []
        #I want 10 nodes  out of this within y
        #10x48
        for i, hidden_bias_weight in enumerate(hidden_bias_weights):
            node = np.dot(hidden_layer, hidden_weights[i]) + (hidden_bias * hidden_bias_weight)

            #y.append(sigmoid(node))
            y.append(softmax(node))


        # print(len(training_weights))
        # print(len(training_weights[0]))
        # print(len(training_inputs))
        # print(len(training_inputs[0]))
        '''
        # Phase 1
        zeta_hidden = np.dot(training_inputs, training_weights) + bias_weight[0]
        hidden_layer = sigmoid(zeta_hidden)

        # Phase 2
        zeta_output = np.dot(hidden_layer, hidden_weights) + hidden_bias_weights[0]
        y = softmax(zeta_output)
        '''

        print(y)


        #t = 1 #ohh t would be the correct answer
        subtracted = np.array(training_outputs) - np.array(y)
        total_network_error = 1/2*(np.square(subtracted))
        #print(sum(total_network_error))


        #back propogation
        #will become for loop with y as an array
        error_of_y = []
        for index, output in enumerate(y):
            error_of_y.append(output*(1-output)*(float(training_outputs[index])-output))
        #error_of_y pops out as: [array([0.03734276])]
        
        error_array = []
        for i, error_in_y in enumerate(error_of_y):
            print(hidden_layer)
            for index, element in enumerate(hidden_layer):
                error_array.append(element*(1-element)*(float(hidden_weights[i][index]) * float(error_in_y)))

        #learn
        #part 1 the hidden layer --> y
        for top_index, y_error in enumerate(error_of_y):
            for index, weight in enumerate(hidden_weights[top_index]):
                weight = weight + learning_rate * y_error * hidden_layer[index]
                #change hidden bias weight
            hidden_bias_weights[top_index] = hidden_bias_weights[top_index] + learning_rate * y_error * hidden_bias
        
        #part 2 the inputs --> hidden layer
        for j, weight in enumerate(training_weights):
            input_index = 0
            for k, variable in enumerate(weight):
                training_weights[j][k] = float(training_weights[j][k]) + (learning_rate * error_array[j] * float(training_inputs[input_index][k]))
            bias_weight[j] = bias_weight[j] + learning_rate * error_array[j] * bias

        print(training_weights)
        print(hidden_weights)
        print(bias_weight)
        print(hidden_bias_weights)

if __name__ == "__main__":

    file = open("digitsTrain.txt", "r")
    contents = file.read()
    lines = re.split(r"\n", contents)

    training_outputs = []
    reading_inputs = [] #read in data
    training_inputs = []

    for line in lines:
        split_line = re.split(r" ", line)
        training_outputs.append(split_line.pop())
        reading_inputs.append(split_line)
    
    #input scrubbing
    for input in reading_inputs:
        input = list(map(int, input))
        input = np.divide(input, 16)
        training_inputs.append(input)
        
    training_outputs = list(map(int, training_outputs))

    num_hidden_nodes = round(len(training_inputs[0]) * .75)

    num_output_nodes = 10

    # number of hidden nodes x number of input nodes 
    # I want length to be the number of input nodes
    training_weights = np.random.rand(num_hidden_nodes, len(training_inputs[0])) # the other example does 64x48
    #training_weights = np.random.rand(len(training_inputs[0]), num_hidden_nodes)
    # print(len(training_weights))
    # print(len(training_weights[0]))
    # number of final outputs x number of hidden nodes
    hidden_weights = np.random.rand(num_output_nodes, num_hidden_nodes) #other example does this opposite
    #hidden_weights = np.random.rand(num_hidden_nodes, num_output_nodes)

    bias = 1
    #len is len(training_weights), no i think it's number of hidden nodes.
    bias_weight = np.random.rand(num_hidden_nodes)
    hidden_bias = 1
    #len is len(final_outputs)
    hidden_bias_weights = np.random.rand(num_output_nodes)

    max_num_epochs = 1
    learning_rate = .1

    neural_net(training_inputs, training_weights, hidden_weights, training_outputs, max_num_epochs,
    learning_rate, bias, bias_weight, hidden_bias, hidden_bias_weights)

    