import numpy as np


class NeuralNet:
    """
    Class representing a neural network and common methods applied to it.
    Properties:
    * weights: 3d array with each sub array taken along the first axis representing a 2d weights arrays between neurons.
        weight[l, i, k] represents the weight between the kth neuron of layer l and the ith neuron of layer l+1
    * biases: 2d array with each sub array representing the biases of each neuron of that layer.
        biases[l, i] represents the bias of neuron i on the layer l.

    Methods:
    * old_compute(a): Computes the result of passing an input layer, a, through the neural network
    * compute(a): Computes the result of passing an input layer or an array of input layers, a, through the neural
        network.
    * feed_forward(a): Computes the weighted sum and activation of each neuron in the neural network given an array of
        input layers, a.
    * stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data): Causes the
        neural network to learn across set number of epochs, given set mini batch size and learning rate. Will
        randomize the inputs given the training data. If given test data against which to compare, will announce the
        percentage of correctly identified testing examples at the end of each epoch, otherwise will announce that
        an epoch has been completed.
    * _learn_mini_batch_(mini_batch, learning_rate): (private method) Adjusts the weights and biases in the neural
        network given the consumes mini_batch and chosen learning rate.
    * _backpropagation_(inputs, labels, learning_rate): (private method) Computes the desired change in each weight
        and bias to maximize learning according to the backpropagation algorithm.
    * evaluate(test_data): Returns the number of correctly computed inputs in test_data.
    * identify_wrong_guesses(test_data): Returns the indices of incorrectly computed inputs in test_data.
    * save(filename): Saves the neural networks weights and biases to filename.
    * load(filename): Loads the neural networks weights and biases to filename.
    """
    def __init__(self, sizes):
        """
        Initialize a neural network given the input sizes, an array of integers each integer corresponding to the
        number of neurons in a given layer. Neural network will have n layers, where n is the length of the array.
        First integer must be equal to the number of input neurons, last must be equal to the number of output neurons.
        """
        sizes = sizes
        self.weights = [np.random.randn(*i) for i in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(i) for i in sizes[1:]]

    @staticmethod
    def cost_prime(outputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Derivative of cost function, used for later calculations.
        """
        y = np.zeros(outputs.shape).reshape(-1, 10)  # Generate zeros array of same shape as output
        np.put_along_axis(y, labels.reshape(-1, 1), 1, axis=1)  # Place a 1 at the label index of each array of y
        return outputs - y

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid function, used for later calculations.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid function, used for later calculations.
        """
        return np.exp(x) / ((np.exp(x) + 1) * (np.exp(x) + 1))

    def old_compute(self, a: np.ndarray) -> np.ndarray:
        """
        Old method for computing output of neural network given an input layer a.
        """
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def compute(self, a: np.ndarray) -> np.ndarray:
        """
        New method for computing output of neural network. a is either an input layer, or an array of input layers.
        """
        if len(a.shape) > 1:
            for w, b in zip(self.weights, self.biases):
                a = self.sigmoid(np.dot(w, a.transpose()) + b.reshape(-1, 1)).transpose()
        else:
            for w, b in zip(self.weights, self.biases):
                a = self.sigmoid(np.dot(w, a) + b)
        return a

    def feedforward(self, a: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Consumes an array of input layers, a, and returns the weighted sums of each neuron given each input layer and
        the activations of each neuron given each input layer. For example, feedforward(a)[0, l, m, i] is the weighted
        sum of neuron i of the first layer, given the mth input layer provided.
        """
        weighted_sums = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1].transpose()).transpose() + b
            weighted_sums.append(z)
            activations.append(self.sigmoid(z))
        return weighted_sums, activations

    def stochastic_gradient_descent(self, training_data: np.ndarray, epochs: int, mini_batch_size: int,
                                    learning_rate: float, test_data=None):
        """
        Note: test_data must be of type None, or np.ndarray.
        Applies stochastic gradient descent learning to the neural network given training_data, a specified number of
        epochs, a specified mini-batch size, a specified learning rate, and optionally testing data to track progress.
        * training_data and test_data must be tuples of two arrays, the first being an array with each element being an
            input layer, and the second with each element being a label corresponding to the input layer of the same
            index.
        """
        inputs, labels = training_data  # Split training data into inputs and labels
        for current in range(epochs):
            rng_state = np.random.get_state()
            np.random.shuffle(inputs)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
            # Shuffle inputs and labels while preserving relative ordering
            mini_batches = [(inputs[i:i + mini_batch_size], labels[i:i + mini_batch_size])
                            for i in range(0, len(inputs), mini_batch_size)]
            # Split inputs and labels into mini batches.
            for m in mini_batches:
                self._learn_mini_batch_(m, learning_rate)  # Apply the learning method to each mini batch

            if test_data:  # Either announce accuracy of neural network or announce completion of epoch
                print('Epoch {0}/{1} complete. Accuracy of: {2}%'.format(
                    current, epochs, round(self.evaluate(test_data) / len(test_data[0]) * 100, 2)))
            else:
                print('Epoch {0}/{1} complete.'.format(current + 1, epochs))

    def _learn_mini_batch_(self, mini_batch: (np.ndarray, np.ndarray), learning_rate: float) -> None:
        """
        Consumes a mini batch and a learning rate and adjusts the weights and biases in neural network according to the
        results of backpropagating each input layer in the neural network.
        * mini_batch must be a tuple of two arrays, the first being an array with each element being an input layer,
            and the second with each element being a label corresponding to the input layer of the same index.
        """
        delta_b, delta_w = self._back_propagation_(mini_batch[0], mini_batch[1], learning_rate)
        # Calculate the required change in bias and weight for each neuron
        for l in range(len(delta_b)):  # Apply the change to each neuron
            self.biases[l] -= delta_b[l]
            self.weights[l] -= delta_w[l]

    def _back_propagation_(self, inputs: np.ndarray, labels: np.ndarray,
                           learning_rate: float) -> (np.ndarray, np.ndarray):
        """
        Consumes an array of input layers, inputs, an array of corresponding labels, and a learning rate, and calculates
        the required change in each neuron weights and bias to maximize learning according to the backpropagation
        algorithm.
        """
        weighted_sums, activations = self.feedforward(inputs)  # Calculate the weighted sums and activations
        errors = [np.zeros(i.shape) for i in weighted_sums]
        errors[-1] = self.cost_prime(activations[-1], labels) * self.sigmoid_prime(weighted_sums[-1])
        # Calculate the error in the last layer given by the equation:
        #   error = cost derivative of activation * sigmoid derivative of weighted sum
        for l in range(-2, -len(errors) - 1, -1):
            errors[l] = np.dot(self.weights[l + 1].transpose(), errors[l + 1].transpose()).transpose() \
                        * self.sigmoid_prime(weighted_sums[l])
        # Calculate the error in each other layer given by the equation:
        #   error in layer l = Transposed weights of layer l+1 [vector product] error in layer l+1 [Hadamard product]
        #       sigmoid derivative of weighted sums of layer l.
        delta_b = [i.sum(axis=0) * (learning_rate / len(inputs)) for i in errors]
        # Average out the change in required change in bias given by each training example and multiply by the learning
        # rate. Note that the change in bias of neuron i = error in neuron i
        delta_w = []
        for l in range(len(errors)):
            delta_w.append(sum([np.dot(a.reshape(-1, 1), b.reshape(1, -1))
                                for a, b in zip(errors[l], activations[l])])
                           * (learning_rate / len(inputs)))
        # Average out the change in required change in weights given by each training example and multiply by the
        # learning rate. Note that the change in weight of neuron i in layer l = error in neuron i [matrix product]
        # activations in layer l-1
        return delta_b, delta_w

    def evaluate(self, test_data: (np.ndarray, np.ndarray)) -> int:
        """
        Consumes an array of input layers along with their correctly identified label, test_data, and returns the number
        of those examples the neural network was able to correctly identify.
        * test_data must be a tuple of two arrays, the first being an array with each element being an input layer,
            and the second with each element being a label corresponding to the input layer of the same index.
        """
        return sum([x.argmax() == y for x, y in zip(self.compute(test_data[0]), test_data[1])])

    def identify_wrong_guesses(self, test_data: np.ndarray) -> [int]:
        """
        Consumes an array of input layers along with their correctly identified label, test_data, and returns the
        indices of incorrectly identified examples.
        * test_data must be a tuple of two arrays, the first being an array with each element being an input layer,
            and the second with each element being a label corresponding to the input layer of the same index.
        """
        guesses = [self.compute(x).argmax() == y for x, y in test_data]
        indexes = []
        for i in range(len(guesses)):
            if not guesses[i]:
                indexes.append(i)
        return indexes

    def save(self, filename: str) -> None:
        """
        Function to save a neural networks weights and biases to filename.
        """
        import csv
        import os
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Weights'])
            for layer in self.weights:
                neurons = [neuron.tolist() for neuron in layer]
                writer.writerow(neurons)
            writer.writerow(['Biases'])
            for row in self.biases:
                writer.writerow(row)

    def load(self, filename: str) -> None:
        """
        Function to load a neural networks weights and biases from filename.
        """
        import csv
        import numpy as np

        def weight_reader(input):
            weights = []
            for layer in input:
                weights.append(np.zeros((len(layer), layer[0].count(',') + 1)))
                for i in range(len(layer)):
                    weights[-1][i] = np.array(layer[i][1:-1].split(','))
            return weights

        def bias_reader(input):
            biases = []
            for layer in input:
                biases.append(np.array([float(i) for i in layer]))
            return biases

        with open(filename, 'r', newline='') as f:
            data = list(csv.reader(f))
            self.weights = weight_reader(data[1:data.index(['Biases'])])
            self.biases = bias_reader(data[data.index(['Biases']) + 1:])
