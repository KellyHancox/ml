import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def neural_network(training_inputs, hidden_weights, hidden_bias, output_weights, output_bias, one_hot_labels):
    error_cost = []

    for epoch in range(100):
    ############# feedforward

        # Phase 1
        zeta_hidden = np.dot(training_inputs, hidden_weights) + hidden_bias
        activation_hidden = sigmoid(zeta_hidden)

        # Phase 2
        zeta_output = np.dot(activation_hidden, output_weights) + output_bias
        activation_output = softmax(zeta_output)

    ########## Back Propagation

    ########## Phase 1

        dcost_dzeta_output = activation_output - one_hot_labels
        dzeta_output_doutput_weights = activation_hidden

        dcost_output_weights = np.dot(dzeta_output_doutput_weights.T, dcost_dzeta_output)

        dcost_output_bias = dcost_dzeta_output

    ########## Phases 2

        dzeta_output_dactivation_hidden = output_weights
        dcost_dactivation_hidden = np.dot(dcost_dzeta_output , dzeta_output_dactivation_hidden.T)
        dactivation_hidden_dzeta_hidden = sigmoid_der(zeta_hidden)
        dzeta_hidden_dhidden_weights = training_inputs
        dcost_hidden_weights = np.dot(dzeta_hidden_dhidden_weights.T, dactivation_hidden_dzeta_hidden * dcost_dactivation_hidden)

        dcost_hidden_bias = dcost_dactivation_hidden * dactivation_hidden_dzeta_hidden

        # Update Weights ================

        hidden_weights -= lr * dcost_hidden_weights
        hidden_bias -= lr * dcost_hidden_bias.sum(axis=0)

        output_weights -= lr * dcost_output_weights
        output_bias -= lr * dcost_output_bias.sum(axis=0)

        if epoch % 200 == 0:
            loss = np.sum(-one_hot_labels * np.log(activation_output))
            print('Loss function value: ', loss)
            error_cost.append(loss)


if __name__ == "__main__":

    np.random.seed(42)

    cat_images = np.random.randn(700, 2) + np.array([0, -3])
    mouse_images = np.random.randn(700, 2) + np.array([3, 3])
    dog_images = np.random.randn(700, 2) + np.array([-3, 3])

    # print(cat_images)
    # print(np.array([0,-3]))

    training_inputs = np.vstack([cat_images, mouse_images, dog_images])

    labels = np.array([0]*700 + [1]*700 + [2]*700)

    one_hot_labels = np.zeros((2100, 3))
    # print(one_hot_labels)

    for i in range(2100):
        one_hot_labels[i, labels[i]] = 1

    print("after transform", one_hot_labels)

    plt.figure(figsize=(10,7))
    plt.scatter(training_inputs[:,0], training_inputs[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)
    #plt.show()

    instances = training_inputs.shape[0]
    attributes = training_inputs.shape[1]
    print('instances', instances) #how many (3,000 something)
    print('attributes', attributes) #64

    #print('feature set', training_inputs);
    hidden_nodes = 4
    output_labels = 3

    hidden_weights = np.random.rand(attributes,hidden_nodes)
    hidden_bias = np.random.randn(hidden_nodes)

    output_weights = np.random.rand(hidden_nodes,output_labels)
    output_bias = np.random.randn(output_labels)
    lr = 10e-4

    print('before')
    neural_network(training_inputs, hidden_weights, hidden_bias, output_weights, output_bias, one_hot_labels)