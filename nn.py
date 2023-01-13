
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(x,0)


def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

class Layer:

    def __init__(self, dim, id, act, act_prime, isoutputLayer = False):
        self.weight = 2 * np.random.random(dim) - 1
        self.delta = None
        self.A = None
        self.activation = act
        self.activation_prime = act_prime
        self.isoutputLayer = isoutputLayer
        self.id = id


    def forward(self, x):
        z = np.dot(x, self.weight)
        self.A = self.activation(z)
        self.dZ = np.atleast_2d(self.activation_prime(z))

        return self.A

    def backward(self, y, rightLayer):
        if self.isoutputLayer:
            error = self.A - y
            self.delta = np.atleast_2d(error * self.dZ)
        else:
            self.delta = np.atleast_2d(
                np.dot(rightLayer.delta, rightLayer.weight.T)
                * self.dZ)
        return self.delta

    def update(self, learning_rate, left_a):
        a = np.atleast_2d(left_a)
        d = np.atleast_2d(self.delta)
        ad = a.T.dot(d)
        self.weight -= learning_rate * ad


class NeuralNetwork:

    def __init__(self, layersDim, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime

        self.layers = []
        for i in range(1, len(layersDim) - 1):
            dim = (layersDim[i - 1] + 1, layersDim[i] + 1)
            self.layers.append(Layer(dim, i, self.activation, self.activation_prime))

        dim = (layersDim[i] + 1, layersDim[i + 1])
        self.layers.append(Layer(dim, len(layersDim) - 1, self.activation, self.activation_prime, True))

    def fit(self, X, y, learning_rate=0.1, epochs=10000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)


        for k in range(epochs):


            a=X

            for l in range(len(self.layers)):
                a = self.layers[l].forward(a)


            delta = self.layers[-1].backward(y, None)

            for l in range(len(self.layers) - 2, -1, -1):
                delta = self.layers[l].backward(delta, self.layers[l+1])



            a = X
            for layer in self.layers:
                layer.update(learning_rate, a)
                a = layer.A

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.layers)):
            a = self.layers[l].forward(a)
        return a



if __name__ == '__main__':

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1],
                  [0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1],
                  [0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]
    ])

    y = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]]).T


    # tanh
    nn = NeuralNetwork([2, 3, 5, 8, 1], activation='tanh')

    nn.fit(X, y, learning_rate=0.1, epochs=10000)

    print("\n\nResult with tanh")
    for e in X:
        print(e, nn.predict(e))


    #  sigmoid
    nn = NeuralNetwork([2, 3, 4, 1], activation='sigmoid')

    nn.fit(X, y, learning_rate=0.3, epochs=20000)

    print("\n\nResult with sigmoid")
    for e in X:
        print(e, nn.predict(e))