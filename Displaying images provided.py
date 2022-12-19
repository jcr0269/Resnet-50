import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import pyplot
from keras.datasets import cifar10
from six.moves import cPickle


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
ROWS: int = 10

x = x_train.astype("uint8")

#Rows and then columns below 
fig, axes1 = plt.subplots(ROWS, ROWS, figsize=(10, 10))
for c in range(ROWS):
    for r in range(ROWS):
        i = np.random.choice(range(len(x)))
        axes1[c][r].set_axis_off()
        axes1[c][r].imshow(x[i:i+1][0])
        plt.show()

        