import random
from mnist import MNIST

data = MNIST()
data.displayImage(random.randint(0, data.getSize()))