import mnist 
import matplotlib.pyplot as plt

data = mnist.MNIST()
all_data = [[],[],[],[],[],[],[],[],[],[],[]]

for item in data.mnist_data:
    try:
        all_data[item[0]] += item[1:]
    except:
        all_data[item[0]] = item[1:]

# Create a 2x5 grid for plotting the ten images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.title('Combined Digits ' + str(i))
    plt.imshow(all_data[i].reshape(28,28), cmap='gray')
    plt.axis('off')  # Hide axes
plt.tight_layout()
plt.show()
