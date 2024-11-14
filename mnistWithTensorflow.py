import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras import Model


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

print(x_train[0])

plt.imshow(x_train[0], cmap='gray')
plt.show()

print(y_train[0])

plt.imshow(x_test[0], cmap='gray')
plt.show()

print(y_test[0])


class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = Conv2D(32, kernel_size=3, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

        def call(self, x):
            x1 = self.conv1(x)
            x2 = self.flatten(x1)
            x3 = self.dense1(x2)
            return self.dense2(x3)


