from Network import *
from mlxtend.data import loadlocal_mnist
from time import time

images, labels = loadlocal_mnist(
    images_path='./data set/train images.idx3-ubyte',
    labels_path='./data set/train labels.idx1-ubyte')
t_images, t_labels = loadlocal_mnist(
    images_path='./data set/test images.idx3-ubyte',
    labels_path='./data set/test labels.idx1-ubyte')
images = np.array([i / 255 for i in images])
t_images = np.array([i / 255 for i in t_images])
data_set = (images, labels)
test_set = (t_images, t_labels)

start = time()
nn = NeuralNet([30])
#nn.load('test.csv')
nn.stochastic_gradient_descent(data_set, 10, 10, 3, test_set)
print('Completed in', round((time() - start), 1), 'seconds')
print('Testing accuracy of {0}%'.format(round(nn.evaluate(test_set) / len(test_set[0]) * 100, 2), '%'))
#nn.save('neural net large.csv')