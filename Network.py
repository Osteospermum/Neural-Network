import numpy as np
from typing import Union, Type, Tuple, List, Callable


class CostClass:
    """
    Generic class defining how the neural network measures its cost

    Must necessarily have:
    * static method cost: The cost function
    * static method prime: The derivative of the cost function
    """
    @staticmethod
    def cost(output: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Must work on an array of inputs with corresponding labels, thereby producing an array of costs
        """
        pass

    @staticmethod
    def prime(output: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Must work on an array of inputs with corresponding labels, thereby producing an array of cost derivative values
        """
        pass


class Quadratic(CostClass):
    @staticmethod
    def cost(outputs: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Quadratic cost function, consumes 2d array of outputs and 1d array of expected values
        """
        y = np.zeros(outputs.shape).reshape(-1, 10)  # Generate zeros array of same shape as output
        np.put_along_axis(y, label.reshape(-1, 1), 1, axis=1)  # Place a 1 at the label index of each array of y
        return np.mean((outputs - y) ** 2, axis=1) / 2

    @staticmethod
    def prime(outputs: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Derivative of cost function, consumes 2d array of outputs and 1d array of expected values.
        """
        y = np.zeros(outputs.shape).reshape(-1, 10)
        np.put_along_axis(y, label.reshape(-1, 1), 1, axis=1)
        return outputs - y


class CrossEntropy(CostClass):
    @staticmethod
    def cost(outputs: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Cross entropy cost function, consumes 2d array of outputs and 1d array of expected values.
        """
        y = np.zeros(outputs.shape).reshape(-1, 10)
        np.put_along_axis(y, label.reshape(-1, 1), 1, axis=1)
        return -np.nan_to_num(np.sum(y * np.log(outputs) + (1 - y) * np.log(1 - outputs), axis=1))

    @staticmethod
    def prime(outputs: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Derivative of cross entropy cost function, consumes 2d array of outputs and 1d array of expected values.
        """
        y = np.zeros(outputs.shape).reshape(-1, 10)
        np.put_along_axis(y, label.reshape(-1, 1), 1, axis=1)
        return -(y - outputs) / (outputs * (1 - outputs))


class RegularizationClass:
    """
    Generic class defining how the neural network should regularize its weights. Note that this means regularization
    class is specifically targeted towards weight decay forms of regularization.

    Must necessarily have:
    * method cost_term: Additional term updating the cost function
    * method learning_rule: New learning rule for weights
    """
    def __init__(self, regularization_coefficient: float) -> None:
        """
        Initializer since it is assumed that a regularization coefficient is necessary
        Note: the regularization class must be initialized with a scaled regularization coefficient
            e.g. λ/n
        """
        self.coefficient = regularization_coefficient

    def cost_term(self, weights: np.ndarray, size: int) -> float:
        pass

    def learning_rule(self, weight: float, learning_rate: float, size: int) -> float:
        """
        Must consume a weight, the neural network's learning rate and the size of the training data and produce
        a new weight by which the partial derivative of cost with respect to said weight should be subtracted
        """
        pass


class L1(RegularizationClass):
    def cost_term(self, weights: np.ndarray, size: int) -> float:
        """
        Function for calculating the L1 regularization term in the cost function
        """
        return (self.coefficient / size) * np.sum([abs(i) for i in weights])

    def learning_rule(self, weight: float, learning_rate: float, size: int) -> float:
        """
        Function for calculating the L1 regularized weight before subtracting by the partial derivative
        """
        return weight - np.sign(weight) * (self.coefficient / size) * learning_rate


class L2(RegularizationClass):
    def cost_term(self, weights: np.ndarray, size: int) -> float:
        """
        Function for calculating the L1 regularization term in the cost function
        """
        return (self.coefficient / size) / 2 * sum([np.sum(i * i) for i in weights])

    def learning_rule(self, weight: float, learning_rate: float, size: int) -> float:
        """
        Function for calculating the L2 regularized weight before subtracting by the partial derivative
        """
        return weight * (1 - (self.coefficient / size) * learning_rate)


def small_weight_initializer(neurons: List[int]) -> List[np.ndarray]:
    """
    Randomized weights initializer where weights have mean 0 and standard deviation equal to the number of input neurons
    of their layer.
    """
    return [np.random.normal(0, 1 / np.sqrt(i), (o, i)) for i, o in zip(neurons[:-1], neurons[1:])]


def large_weight_initializer(neurons: List[int]) -> List[np.ndarray]:
    """
    Randomized weights initializer where weights have mean 0 and standard deviation 1.
    """
    return [np.random.randn(*i) for i in zip(neurons[1:], neurons[:-1])]


def gaussian_bias_initializer(neurons: List[int]) -> List[np.ndarray]:
    """
    Randomized biases initializer where biases have mean 0 and standard deviation 1.
    """
    return [np.random.randn(i) for i in neurons[1:]]


def zeros_bias_initializer(neurons: List[int]) -> List[np.ndarray]:
    """
    Biases initializer where all biases are equal to 0.
    """
    return [np.zeros(i) for i in neurons[1:]]


class NeuralNet:
    """
    Class representing a neural network and common methods applied to it.

    Properties:
    * weights: 3d array with each sub array taken along the first axis representing a 2d weights arrays between neurons.
        weight[l, i, k] represents the weight between the kth neuron of layer l and the ith neuron of layer l+1
    * biases: 2d array with each sub array representing the biases of each neuron of that layer.
        biases[l, i] represents the bias of neuron i on the layer l.
    * cost(outputs, label, weights): The cost function by which to measure the neural network.
    * cost_prime(outputs, label): The derivative of the cost function of the neural network.
    * weight_learning_rule(weight, learning_rate, size): A function which is applied to weights before subtracting
        the partial derivative of the cost function with respect to the weight.

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
    def __init__(self,
                 neurons: List[int],
                 cost_class: Type[CostClass] = CrossEntropy,
                 regularization_class: Type[RegularizationClass] = None,
                 weight_initializer: Callable = small_weight_initializer,
                 bias_initializer: Callable = gaussian_bias_initializer) -> None:
        """
        Initialize a neural network given the input sizes, an array of integers each integer corresponding to the
        number of neurons in a given layer. Neural network will have n layers, where n is the length of the array.
        First integer must be equal to the number of input neurons, last must be equal to the number of output neurons.

        Optionally allows specification of cost function, in this case a CostClass subclass must be passed with two
        static methods, cost and prime. See class details for structuring.

        Optionally allows specification of weight decay regularization, in this case a RegularizationClass with two
        methods, cost_term and learning_rule. See class details for structuring.

        Optionally allows specification of weight and bias initialization, each requires a function which consumes a
        list of the number of neurons in each layer and produces weights or biases accordingly.

        Note that the constructor will likely throw a warning when provided with an instance of a subclass of
        RegularizationClass. There is no way to change this without including an external library and I didn't think
        it was of enough importance to warrant forcing an additional library install.
        """
        self.weights = weight_initializer(neurons)
        self.biases = bias_initializer(neurons)
        if regularization_class:  # If regularization is being used
            self.cost = lambda outputs, label, weights, size: cost_class.cost(outputs, label) + \
                                                        regularization_class.cost_term(weights, size)
            self.cost_prime = cost_class.prime
            self.weight_learning_rule = regularization_class.learning_rule
        else:  # If regularization isn't being used
            self.cost = lambda outputs, label, weights, size: cost_class.cost(outputs, weights)
            self.cost_prime = cost_class.prime
            self.weight_learning_rule = lambda weight, learning_rate, size: weight

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

    def feedforward(self, a: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    def stochastic_gradient_descent(self, training_data: Tuple[np.ndarray, np.ndarray], epochs: int,
                                    mini_batch_size: int, learning_rate: float,
                                    test_data: Union[None, Tuple[np.ndarray, np.ndarray]] = None,
                                    verbose: bool = False) -> None:
        """
        Applies stochastic gradient descent learning to the neural network given training_data, a specified number of
        epochs, a specified mini-batch size, a specified learning rate, and optionally testing data to track progress.
        * training_data and test_data must be tuples of two arrays, the first being an array with each element being an
            input layer, and the second with each element being a label corresponding to the input layer of the same
            index.

        Optionally verbose mode can be used, however it requires testing data is provided
        """
        if verbose:
            if not training_data:
                raise Exception('Verbose mode requires testing data')

            from time import time
            test_inputs, test_labels = test_data
            training_costs, training_accuracies = [], []
            testing_costs, testing_accuracies = [], []

        inputs, labels = training_data  # Split training data into inputs and labels
        size = len(inputs)  # Size of training data
        for current in range(epochs):
            if verbose:
                start = time()
            rng_state = np.random.get_state()
            np.random.shuffle(inputs)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
            # Shuffle inputs and labels while preserving relative ordering
            mini_batches = [(inputs[i:i + mini_batch_size], labels[i:i + mini_batch_size])
                            for i in range(0, len(inputs), mini_batch_size)]
            # Split inputs and labels into mini batches.
            for m in mini_batches:
                self._learn_mini_batch_(m, learning_rate, size)  # Apply the learning method to each mini batch

            if verbose:  # End of epoch announcements
                training_costs.append(np.sum(self.cost(self.compute(inputs), labels, self.weights, size)))
                training_accuracies.append(self.evaluate(training_data) / size * 100)
                testing_costs.append(np.sum(self.cost(
                    self.compute(test_inputs), test_labels, self.weights, len(test_inputs))))
                testing_accuracies.append(self.evaluate(test_data) / len(test_inputs) * 100)
                print(str('Epoch {0}/{1} complete. Training accuracy: {2}%, Training cost: {3}, ' +
                          'Average training Cost: {4}, Testing accuracy: {5}%, Testing cost: {6} ' +
                          'Average testing cost: {7}, Completed in {8}s').
                      format(current + 1, epochs, round(training_accuracies[-1], 3), round(training_costs[-1], 3),
                             round(training_costs[-1]/len(inputs), 3), round(testing_accuracies[-1], 3),
                             round(testing_costs[-1], 3), round(testing_costs[-1]/len(test_inputs), 3),
                             round((time() - start), 3)))
            elif test_data:
                print('Epoch {0}/{1} complete. Accuracy of: {2}%'.format(
                    current+1, epochs, round(self.evaluate(test_data) / len(test_data[0]) * 100, 2)))
            else:
                print('Epoch {0}/{1} complete.'.format(current + 1, epochs))

        if verbose:  # Return metrics
            return training_costs, training_accuracies, testing_costs, testing_accuracies

    def _learn_mini_batch_(self, mini_batch: Tuple[np.ndarray, np.ndarray], learning_rate: float, size: int) -> None:
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
            self.weights[l] = self.weight_learning_rule(self.weights[l], learning_rate, size) - delta_w[l]

    def _back_propagation_(self, inputs: np.ndarray, labels: np.ndarray,
                           learning_rate: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> int:
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
        Function to save a neural networks weights, biases, cost functions, regularization, etc. to a file.

        Note that to save the neural network a third party library, dill, is required as there is no straightforward way
        to save a class to a file, while including the lambda cost function.
        """
        import dill
        dill.dump(self, open(filename, "wb"))
        print("Successfully saved to {0}".format(filename))


def load(filename: str) -> NeuralNet:
    """
    Function to load and subsequently return an instance of a neural network class.

    Note that to save the neural network a third party library, dill, is required as there is no straightforward way
    to save a class to a file, while including the lambda cost function.
    """
    import dill
    net = dill.load(open(filename, "rb"))
    print("Successfully loaded from {0}".format(filename))
    return net
