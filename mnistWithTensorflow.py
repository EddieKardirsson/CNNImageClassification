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


loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

model = MNISTModel()


@tf.function
def train_step(inputs, outputs):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_function(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(outputs, predictions)


@tf.function
def test_step(inputs, outputs):
    predictions = model(inputs)
    loss = loss_function(outputs, predictions)

    test_loss(loss)
    test_accuracy(outputs, predictions)


x_train, x_test = x_train / 255.0, x_test / 255.0

x_train =x_train[..., tf.newaxis]
x_test =x_test[..., tf.newaxis]

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)