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


class SMOTEDataAugmentationLayer:
    def __init__(self, kNeighbors=5, noiseLevel=0.1):
        self.kNeighbors = kNeighbors
        self.noiseLevel = noiseLevel

    def forward(self, X, augment=True):
        if not augment:
            return X

        augmentedSamples = []
        for i in range(X.shape[0]):
            # Randomly choose augmentation technique
            choice = np.random.choice(['smote', 'noise', 'flip'])

            if choice == 'smote':
                # For SMOTE
                neighbors = self.findKNeighbors(X, X[i])
                chosenNeighbor = neighbors[np.random.choice(self.kNeighbors)]
                alpha = np.random.random()
                syntheticSample = self.interpolate(X[i], chosenNeighbor, alpha)
                augmentedSamples.append(syntheticSample)
            elif choice == 'noise':
                # Add noise
                noisyImage = self.addNoise(X[i])
                augmentedSamples.append(noisyImage)
            else:
                # Reflect image
                flippedImage = self.horizontalFlip(X[i])
                augmentedSamples.append(flippedImage)

        return np.vstack(augmentedSamples)

    def findKNeighbors(self, data, sample):
        # Here's a basic idea using Euclidean distance.
        distances = np.linalg.norm(data - sample, axis=1)
        neighborsIdx = np.argsort(distances)[:self.kNeighbors]
        return data[neighborsIdx]

    def interpolate(self, image1, image2, alpha):
        return alpha * image1 + (1 - alpha) * image2

    def addNoise(self, image):
        noise = np.random.normal(0, self.noiseLevel, image.shape)

        # Check if the image is normalized
        if np.max(image) <= 1:
            return np.clip(image + noise, 0, 1)
        else:
            return np.clip(image + noise, 0, 255)

    def horizontalFlip(self, image):
        isFlattened = len(image.shape) == 1
        if isFlattened:
            sideLength = int(np.sqrt(len(image)))
            image = image.reshape(sideLength, sideLength)

        rows, cols = image.shape
        flippedImage = np.zeros_like(image)
        for row in range(rows):
            flippedImage[row, :] = image[row, ::-1]

        if isFlattened:
            flippedImage = flippedImage.flatten()

        return flippedImage

    def backward(self, dZ):
        return dZ


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
        sigmoid_output = 1 / (1 + np.exp(-dataIn))
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
    def __init__(self, sizeIn, sizeOut, initialization="random", optimizer="adam"):
        super().__init__()
        self.train_input = None
        # Weights and Biases Initialization
        if initialization == "xavier":
            limit = np.sqrt(6.0 / (sizeIn + sizeOut))
            self.__weights = np.random.uniform(-limit, limit, (sizeIn, sizeOut))
        elif initialization == "he":
            stddev = np.sqrt(2.0 / sizeIn)
            self.__weights = np.random.normal(0, stddev, (sizeIn, sizeOut))
        else:  # Default random initialization
            self.__weights = np.random.uniform(-0.0001, 0.0001, (sizeIn, sizeOut))

        self.__biases = np.zeros((1, sizeOut))

        # Optimization Parameters
        self.optimizer = optimizer
        self.learning_rate = 0.0001

        # Adam Parameters Initialization
        if self.optimizer == "adam":
            self.rho1 = 0.9
            self.rho2 = 0.999
            self.eta = 0.001
            self.delta = 1e-8
            self.s_weight = np.zeros((sizeIn, sizeOut))
            self.s_bias = np.zeros((1, sizeOut))
            self.r_weight = np.zeros((sizeIn, sizeOut))
            self.r_bias = np.zeros((1, sizeOut))
            self.t = 0  # initialize time step

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
        self.train_input = np.array(dataIn)
        return output_data

    # Input : None
    # Output : Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        return np.array(self.__weights).T

    def updateWeightsWithAdam(self, gradW, gradB):
        self.t += 1  # increment time step

        # Update the first moment (like momentum) for weights and biases
        self.s_weight = self.rho1 * self.s_weight + (1 - self.rho1) * gradW
        self.s_bias = self.rho1 * self.s_bias + (1 - self.rho1) * gradB

        # Update the second moment (like RMSProp) for weights and biases
        self.r_weight = self.rho2 * self.r_weight + (1 - self.rho2) * (gradW ** 2)
        self.r_bias = self.rho2 * self.r_bias + (1 - self.rho2) * (gradB ** 2)

        # Correct bias for the first and second moments (bias correction)
        s_weight_corr = self.s_weight / (1 - self.rho1 ** self.t)
        s_bias_corr = self.s_bias / (1 - self.rho1 ** self.t)

        r_weight_corr = self.r_weight / (1 - self.rho2 ** self.t)
        r_bias_corr = self.r_bias / (1 - self.rho2 ** self.t)

        # Update weights and biases using the Adam optimization algorithm
        self.__weights -= self.eta * s_weight_corr / (np.sqrt(r_weight_corr) + self.delta)
        self.__biases -= self.eta * s_bias_corr / (np.sqrt(r_bias_corr) + self.delta)

    def backward(self, gradIn):
        # compute the gradient with respect to weights
        gradW = self.train_input.T @ gradIn / len(self.train_input)
        # compute the gradient with respect to biases
        gradB = np.sum(gradIn, axis=0, keepdims=True)

        # Update weights and biases
        if self.optimizer == "adam":
            self.updateWeightsWithAdam(gradW, gradB)
        else:
            self.__weights -= self.learning_rate * gradW
            self.__biases -= self.learning_rate * gradB

        # compute and return the gradient with respect to input
        gradOut = gradIn @ np.array(self.gradient())  # Using matrix multiplication (@) instead of dot product
        return gradOut

    def updateWeights(self, gradIn, eta=0.0001):

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


class DropoutLayer:
    def __init__(self, dropoutProb):
        self.dropoutProb = dropoutProb
        self.trainMode = True  # This helps in differentiating between training and testing
        self.mask = None  # This will store which neurons are 'dropped out'

    def forward(self, inputData):
        # Performs the forward pass of the dropout layer.
        if self.trainMode:
            # Generate a binary mask indicating which neurons are dropped
            self.mask = np.random.binomial(1, 1 - self.dropoutProb, size=inputData.shape) / (1.0 - self.dropoutProb)
            return inputData * self.mask
        else:
            # If it's test/validate mode, don't drop any neuron
            return inputData

    def backward(self, gradOutput):
        # Performs the backward pass of the dropout layer.
        if self.trainMode:
            # Only propagate gradient to the non-dropped neurons
            return gradOutput * self.mask
        else:
            return gradOutput

    def setTrainMode(self):
        # Set the layer into training mode.
        self.trainMode = True

    def setTestMode(self):
        # Set the layer into test/evaluation mode.
        self.trainMode = False


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
