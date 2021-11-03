import random
import tensorflow as tf
from scipy import ndimage
import numpy as np
import skimage.exposure as ske
import warnings

class Augmentation_3D:
    def __init__(self, transformations):
        self.transformations = transformations

    class CallByName:
        def get_method(self, method_name, volume):
            method = getattr(self, method_name)(volume)
            return method

        @tf.function
        def rotate(self, volume):
            """Rotate the volume by a few degrees"""

            def scipy_rotate(volume):
                # define some rotation angles
                angles = [-30, -20, -10, 10, 20, 30]
                # pick angles at random
                angle = random.choice(angles)
                # rotate volume
                volume = ndimage.rotate(volume, angle, reshape=False)
                return volume

            augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float64)
            return augmented_volume

        @tf.function
        def flip_horizontally(self, volume):
            """Flip horizontally the volume"""

            def numpy_flip_horizontally(volume):
                # flip volume
                volume = np.flip(volume, axis=1)
                return volume

            augmented_volume = tf.numpy_function(numpy_flip_horizontally, [volume], tf.float64)
            return augmented_volume

        @tf.function
        def flip_vertically(self, volume):
            """Flip vertically the volume"""

            def numpy_flip_vertically(volume):
                # flip volume
                volume = np.flip(volume, axis=0)
                return volume

            augmented_volume = tf.numpy_function(numpy_flip_vertically, [volume], tf.float64)
            return augmented_volume

        @tf.function
        def brightness(self, volume):
            """Adjust the exposure of the volume"""

            def skimage_brightness(volume):
                gamma = np.random.uniform(0.4, 1)
                gain = np.random.uniform(0, 1)
                volume = ske.adjust_gamma(volume, gamma, gain)

                return volume

            augmented_volume = tf.numpy_function(skimage_brightness, [volume], tf.float64)
            return augmented_volume

    def train_preprocessing(self, volume, label):
        """
        Process training data by transforming and adding channel.
        :param volume: image volume
        :param label: image label
        :return: transformed volume
        """

        # apply the available augmentations
        call_by_name = self.CallByName()
        random_aug_name = random.choice(self.transformations)
        try:
            aug_func = call_by_name.get_method(random_aug_name, volume)
        except:
            raise ValueError('The transformation is not supported! Modify the augmenration_3D.py to add your arbitrary transformations')
        transformed_volume = aug_func

        # expand the dimensions to be useful for the model
        transformed_volume = tf.expand_dims(transformed_volume, axis=3)

        return transformed_volume, label

    def validation_preprocessing(self, volume, label):
        """
        Process training data by only adding a channel.
        :param volume: image volume
        :param label: image label
        :return: transformed volume
        """
        volume = tf.expand_dims(volume, axis=3)

        return volume, label





