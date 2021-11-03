from keras.models import Model, Input
from keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPooling3D, ReLU, BatchNormalization
import tensorflow as tf

class CNN3D(tf.keras.models.Model):
    def __init__(self, img_size=(128, 128), training=True):
        super(CNN3D, self).__init__()
        self.img_size = img_size
        self.training = training
        self.model = None

    def build(self):
        input = Input(shape=(self.img_size[0], self.img_size[1], 3, 1), name='input')
        X = Conv3D(filters=16, kernel_size=(7, 7, 1), strides=(1, 1, 1), padding='valid', name='conv1',
                       kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None))(input)
        X = ReLU(name="activation_conv1")(X)

        X = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='valid', name='pool1')(X)
        X = BatchNormalization()(X)

        X = Conv3D(filters=16, kernel_size=(13, 13, 2), strides=(1, 1, 1), padding='valid', name='conv2',
                       kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None))(X)
        X = ReLU(name="activation_conv2")(X)

        X = MaxPooling3D(pool_size=(3, 3, 1), strides=(1, 1, 1), padding='valid', name='pool2')(X)
        x = BatchNormalization()(X)

        X = Conv3D(filters=64, kernel_size=(10, 10, 1), strides=(1, 1, 1), padding='valid', name='conv3',
                       kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None))(X)
        X = ReLU(name="activation_conv3")(X)

        X = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1), name='pool3')(X)
        X = BatchNormalization()(X)

        X = Flatten()(X)
        X = Dense(units=64, name='fc1')(X)
        X = tf.keras.activations.relu(X)
        if self.training:
            X = Dropout(0.1)(X)

        X = Dense(units=4, name='fc2')(X)
        X = tf.keras.activations.relu(X)
        if self.training:
            X = Dropout(0.1)(X)

        X = Dense(units=1, activation='sigmoid', name='FC_F')(X)
        model = Model(inputs=input, outputs=X)
        self.model = model

        return model
