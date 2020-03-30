import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x + (1 - x)


training_inputs = np.array([[0, 0, 1],[1, 1, 1],[1, 0, 1],[0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1) #random numbers

synaptic_weights = 2 * np.random.random((3, 1)) -1 #3 by 1 matrix bc 3 input 1 output
print('random starting synaptic weights')
print(synaptic_weights)

for epoch in range(50000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights)) #product of matrices

    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments) #transpose so it matches

print('synaptic weights after training')
print(synaptic_weights)

print('outputs after training')
print(outputs)