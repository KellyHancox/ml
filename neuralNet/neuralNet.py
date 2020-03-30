import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    #putting feed forward and back propogation together
    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):
            #thinking is getting dot product between inputs and weights
            #outputs the sigmoid of these
            output = self.think(training_inputs)
            print("outputs:")
            print(output)

            #nget difference of error
            error = training_outputs - output
            print("error: ")
            print(error)

            #matrix multiplication
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments
    
    def think(self, inputs):
        #using dot product again. our inputs are integers, our synaptic weights are floats
        #so we'll have to change inputs to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

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
    training_inputs = np.array([[0,0,0,0],
                                [1,0,0,0],
                                [0,0,0,1],
                                [0,1,0,2],
                                [0,2,1,2],
                                [1,2,1,2],
                                [1,2,1,0],
                                [0,1,0,0],
                                [0,2,1,0],
                                [0,1,1,2],
                                [1,1,1,0],
                                [1,1,0,0],
                                [0,0,1,0],
                                [1,1,0,2]])
    training_outputs = np.array([[1,0,1,1,0,0,0,1,1,0,1,1,1,0]]).T

    #training iterations are a custom input
    neural_network.train(training_inputs, training_outputs, 2)

    print("syn weights after training: ")
    print(neural_network.synaptic_weights)

    '''
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    D = str(input("Input 4: "))

    print("new situation: input data = ", A, B, C, D)
    print("output data: ")
    print(neural_network.think(np.array([A, B, C, D])))
    '''