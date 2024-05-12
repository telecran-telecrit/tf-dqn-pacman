import tensorflow as tf
from tensorflow import keras, optimizers, layers
from tensorflow.keras import layers

class DQN(keras.Model):

    CONV_N_MAPS = [4, 32, 32]
    CONV_KERNEL_SIZES = [(4, 4), (2, 2)]
    CONV_STRIDES = [2, 2]
    CONV_PADDINGS = [2, 0]
    N_HIDDEN_IN = 32 * 11 * 10
    N_HIDDEN = [512, 128]

    def __init__(self, outputs, **kwargs):
        super(DQN, self).__init__(**kwargs)
        conv2d = lambda i: layers.Conv2D(
            self.CONV_N_MAPS[i],
            self.CONV_KERNEL_SIZES[i],
            strides=self.CONV_STRIDES[i],
            padding="valid",
            activation="relu",
        )
        self.conv1 = conv2d(0)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv2d(1)
        self.bn2 = layers.BatchNormalization()

        self.hidden1 = layers.Dense(self.N_HIDDEN[0], activation="relu")
        self.hidden2 = layers.Dense(self.N_HIDDEN[1], activation="relu")
        self.output = layers.Dense(outputs)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = layers.Flatten()(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.output(x)
