import pandas as pd
import matplotlib.pyplot as plt
import random

mnist_data = pd.read_csv('data/mnist.csv').values

labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]

SELECTED_NUMBER = random.randint(0, len(labels))
img_size = 28

plt.title('Label is {label}'.format(label=labels[SELECTED_NUMBER]))
plt.imshow(digits[SELECTED_NUMBER].reshape(img_size, img_size))
plt.show()