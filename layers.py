import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward(self, gradIn):
        pass


class InputLayer(Layer):
    # Input : dataIn , an NxD matrix
    # Output : None
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)  # axis=0 operations (like np.mean, np.sum, etc.) will be performed
        # vertically, down the rows.
        self.stdX = np.std(dataIn, axis=0, ddof=1)  # ddof=1 the divisor becomes N - ddof, or N - 1
        self.stdX[self.stdX == 0] = 1  # To avoid divide-by-zero issues

    # Input : dataIn , an NxD matrix
    # Output : An NxD matrix
    def forward(self, dataIn):
        # Compute z-score of input data
        zscored_data = (dataIn - self.meanX) / self.stdX
        # Store the input and output data for later use
        self.setPrevIn(dataIn)
        self.setPrevOut(zscored_data)
        return zscored_data

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass


class LinearLayer(Layer):
    # Input : None
    # Output : None
    def __init__(self):
        super().__init__()

    # Input : dataIn , an NxK matrix
    # Output : An NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    # Input : None
    # Output : Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        N, D = np.array(self.getPrevOut()).shape
        return np.array([np.eye(D) for _ in range(N)])

    def backward(self, gradIn):
        return gradIn


class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        relu_output = np.maximum(0, dataIn)
        self.setPrevIn(dataIn)
        self.setPrevOut(relu_output)
        return relu_output

    # Input : None
    # Output : Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        # Derivative of ReLu function:
        # 1 for 𝑧_j >= 0
        # 0 for 𝑧_j < 0
        return (np.array(self.getPrevOut()) >= 0).astype(int)

    def backward(self, gradIn):
        relu_grad = self.gradient()
        return gradIn * relu_grad


class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        clipped_data = np.clip(dataIn, -500, 500)  # Clip values to avoid overflow
        sigmoid_output = 1 / (1 + np.exp(-clipped_data))
        self.setPrevOut(sigmoid_output)
        return sigmoid_output

    # Input : None
    # Output : Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        # Gradient of the sigmoid function is sigmoid * (1 - sigmoid)
        sigmoid_output = self.getPrevOut()
        return sigmoid_output * (1 - np.array(sigmoid_output))  # * Numpy performs Hadamard Product multiplication

    def backward(self, gradIn):
        sigmoid_grad = self.gradient()
        return gradIn * sigmoid_grad


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

        self.train_output = None

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        data_max_subtracted = dataIn - np.max(dataIn, axis=1, keepdims=True)
        exps = np.exp(data_max_subtracted)
        softmax_output = exps / np.sum(exps, axis=1, keepdims=True)
        self.setPrevOut(softmax_output)
        self.train_output = softmax_output
        return softmax_output

    # Input : None
    # Output : Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        # Compute the softmax output
        softmax_output = np.array(self.train_output)

        # Initialize a 3D tensor to hold the gradients
        N, K = np.array(softmax_output).shape
        jacobian = np.zeros((N, K, K))

        # Compute the Jacobian matrix for each instance
        for n in range(N):
            for i in range(K):
                for j in range(K):
                    if i == j:
                        # Computation for elements on the diagonal
                        jacobian[n, i, j] = softmax_output[n, i] * (1 - softmax_output[n, j])
                    else:
                        # Computation for off-diagonal elements
                        jacobian[n, i, j] = -softmax_output[n, i] * softmax_output[n, j]

        return jacobian

    def backward(self, gradIn):
        jacobian = self.gradient()
        jacobian = jacobian[:gradIn.shape[0]]
        return np.einsum('...i,...ij->...j', gradIn, jacobian)


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        # Explicitly computing the tanh function
        exp_data = np.exp(dataIn)
        exp_neg_data = np.exp(-dataIn)
        tanh_output = (exp_data - exp_neg_data) / (exp_data + exp_neg_data)
        self.setPrevOut(tanh_output)
        return tanh_output

    # Input : None
    # Output : Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        # Gradient is 1 - (1 - 𝑔_𝑗^2(𝒛)), here prevOut is tanh_output
        tanh_grad = 1 - np.power(self.getPrevOut(), 2)
        return tanh_grad

    def backward(self, gradIn):
        tanh_grad = self.gradient()
        return gradIn * tanh_grad


class FullyConnectedLayer(Layer):
    # Input : sizeIn, the number of features of data coming in
    # Input : sizeOut, the number of features for the data coming out.
    # Output : None
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        # Initialize weights and biases with small random values like plus minus 10^{-4}
        self.learning_rate = 0.0001
        self.__weights = np.random.uniform(-0.0001, 0.0001, (sizeIn, sizeOut))
        self.__biases = np.random.uniform(-0.0001, 0.0001, (1, sizeOut))

    # Input : None
    # Output : The sizeIn x sizeOut weight matrix.
    def getWeights(self):
        return self.__weights

    # Input : The sizeIn x sizeOut weight matrix.
    # Output : None
    def setWeights(self, weights):
        self.__weights = weights

    # Input : The 1 x sizeOut bias vector
    # Output : None
    def getBiases(self):
        return self.__biases

    # Input : None
    # Output : The 1 x sizeOut bias vector
    def setBiases(self, biases):
        self.__biases = biases

    # Input : dataIn, an NxD data matrix
    # Output : An NxK data matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output_data = np.matmul(dataIn, self.__weights) + self.__biases
        self.setPrevOut(output_data)
        return output_data

    # Input : None
    # Output : Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        return np.array(self.__weights).T

    def backward(self, gradIn):
        # compute the gradient with respect to weights
        gradW = np.mean(np.array(self.getPrevIn()).T @ gradIn, axis=1, keepdims=True)

        # compute the gradient with respect to biases
        gradB = np.sum(gradIn, axis=0, keepdims=True)

        # update weights and biases
        self.__weights -= self.learning_rate * gradW
        self.__biases -= self.learning_rate * gradB

        # compute and return the gradient with respect to input
        gradOut = gradIn @ np.array(self.gradient())  # Using matrix multiplication (@) instead of dot product
        return gradOut

    def updateWeights(self, gradIn, eta = 0.0001):

        dJdb = np.sum(gradIn, axis=0, keepdims=True) / gradIn.shape[0]
        dJdW = np.sum((np.array(self.getPrevIn()).T @ gradIn) / gradIn.shape[0], axis=1, keepdims=True)

        if dJdW.shape != self.__weights.shape:
            # reshape dJdW to match with self.__weights
            dJdW = dJdW.reshape(self.__weights.shape)

        self.__weights -= eta * dJdW

        # Ensure the bias update term is of the correct shape
        if dJdb.shape != self.__biases.shape:
            dJdb = np.mean(dJdb)

        self.__biases -= eta * dJdb


class SquaredError():
    def __init__(self):
        super().__init__()

    # Input : Y is an N by K matrix of target values.
    # Input : Yhat is an N by K matrix of estimated values.
    # Output : A single floating point value.
    def eval(self, Y, Yhat):
        return np.mean((Y - Yhat) * (Y - Yhat))

    # Input : Y is an N by K matrix of target values.
    # Input : Yhat is an N by K matrix of estimated values.
    # Output : An N by K matrix.
    def gradient(self, Y, Yhat):
        return -2 * (Y - Yhat)


class LogLoss():
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    # Input : Y is an N by K matrix of target values.
    # Input : Yhat is an N by K matrix of estimated values.
    # Output : A single floating point value.
    def eval(self, Y, Yhat):
        Yhat = np.clip(Yhat, self.eps, 1 - self.eps)
        return - np.mean(Y * np.log(Yhat + self.eps) + (1 - Y) * np.log(1 - Yhat + self.eps))

    # Input : Y is an N by K matrix of target values.
    # Input : Yhat is an N by K matrix of estimated values.
    # Output : An N by K matrix.
    def gradient(self, Y, Yhat):
        Yhat = np.clip(Yhat, self.eps, 1 - self.eps)
        return - (Y - Yhat) / (Yhat * (1 - Yhat) + self.eps)


class CrossEntropy():
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    # Input : Y is an N by K matrix of target values.
    # Input : Yhat is an N by K matrix of estimated values.
    # Output : A single floating point value.
    def eval(self, Y, Yhat):
        Yhat = np.clip(Yhat, self.eps, 1 - self.eps)
        return - np.mean(Y * np.log(Yhat + self.eps))

    # Input : Y is an N by K matrix of target values.
    # Input : Yhat is an N by K matrix of estimated values.
    # Output : An N by K matrix.
    def gradient(self, Y, Yhat):
        Yhat = np.clip(Yhat, self.eps, 1 - self.eps)
        return - (Y / (Yhat + self.eps))
