import numpy as np


class NeuralNet:
    def __init__(self, sizes):
        sizes = [784] + sizes + [10]
        self.weights = [np.random.randn(*i) for i in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(i) for i in sizes[1:]]

    @staticmethod
    def cost(output, label):
        y = np.zeros(10)
        y[label] = 1
        return np.mean([i * i for i in y - output]) / 2

    @staticmethod
    def cost_prime(output, label):
        y = np.zeros(output.shape).reshape(-1, 10)
        np.put_along_axis(y, label.reshape(-1, 1), 1, axis=1)
        return output - y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return np.exp(x) / ((np.exp(x) + 1) * (np.exp(x) + 1))

    def old_compute(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def compute(self, a):
        if len(a.shape) > 1:
            for w, b in zip(self.weights, self.biases):
                a = self.sigmoid(np.dot(w, a.transpose()) + b.reshape(-1, 1)).transpose()
        else:
            for w, b in zip(self.weights, self.biases):
                a = self.sigmoid(np.dot(w, a) + b)
        return a

    def feedforward(self, a):
        weighted_sums = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1].transpose()).transpose() + b
            weighted_sums.append(z)
            activations.append(self.sigmoid(z))
        return weighted_sums, activations

    def evaluate(self, test_data):
        return sum([x.argmax() == y for x, y in zip(self.compute(test_data[0]), test_data[1])])

    def identify_wrong_guesses(self, test_data):
        guesses = [self.compute(x).argmax() == y for x, y in test_data]
        indexes = []
        for i in range(len(guesses)):
            if not guesses[i]:
                indexes.append(i)
        return indexes

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        for current in range(epochs):
            inputs, labels = training_data
            rng_state = np.random.get_state()
            np.random.shuffle(inputs)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
            mini_batches = [(inputs[i:i + mini_batch_size], labels[i:i + mini_batch_size])
                            for i in range(0, len(inputs), mini_batch_size)]
            for m in mini_batches:
                self.learn_mini_batch(m, learning_rate)

            if test_data:
                print('Epoch {0} complete. Accuracy of: {1}%'.format(
                    current, round(self.evaluate(test_data) / len(test_data[0]) * 100, 2), '%'))
            else:
                print('Epoch {0} complete.'.format(current))

    def learn_mini_batch(self, mini_batch, learning_rate):
        delta_b, delta_w = self.back_propagation(mini_batch[0], mini_batch[1], learning_rate)
        for l in range(len(delta_b)):
            self.biases[l] -= delta_b[l]
            self.weights[l] -= delta_w[l]

    def back_propagation(self, inputs, labels, learning_rate):
        weighted_sums, activations = self.feedforward(inputs)
        errors = [np.zeros(i.shape) for i in weighted_sums]
        errors[-1] = self.cost_prime(activations[-1], labels) * self.sigmoid_prime(weighted_sums[-1])
        for l in range(-2, -len(errors) - 1, -1):
            errors[l] = np.dot(self.weights[l + 1].transpose(), errors[l + 1].transpose()).transpose() \
                        * self.sigmoid_prime(weighted_sums[l])
        delta_b = [i.sum(axis=0) * (learning_rate / len(inputs)) for i in errors]
        delta_w = []
        for l in range(len(errors)):
            delta_w.append(sum([np.dot(a.reshape(-1, 1), b.reshape(1, -1))
                                for a, b in zip(errors[l], activations[l])])
                           * (learning_rate / len(inputs)))
        return delta_b, delta_w

    def save(self, filename):
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

    def load(self, filename):
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
