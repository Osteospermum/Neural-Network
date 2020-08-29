import sys
from Network import *

nn = NeuralNet([16])
nn.load('neural net.csv')
current_image = np.array([int(i) / 255 for i in sys.argv[1].split(',')])
current_label = int(sys.argv[2])
results = nn.compute(current_image)
print(results.argmax())
print(round(max(results), 3))
print(round(results[current_label], 3))
