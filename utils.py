import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plot_single_image(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

def plot_volume(volume):
    fig = plt.figure()
    # create a 128 x 128 vertex mesh
    xx, yy = np.meshgrid(np.linspace(0, 10, 128), np.linspace(0, 10, 128))
    zz = 10 * np.ones(xx.shape)

    # show the reference image
    ax1 = fig.add_subplot(221)
    ax1.imshow(volume[:, :, 0], cmap=plt.get_cmap('gray'), interpolation='nearest', extent=[0, 10, 0, 10])

    ax2 = fig.add_subplot(222)
    ax2.imshow(volume[:, :, 1], cmap=plt.get_cmap('gray'), interpolation='nearest', extent=[0, 10, 0, 10])

    ax3 = fig.add_subplot(223)
    ax3.imshow(volume[:, :, 2], cmap=plt.get_cmap('gray'), interpolation='nearest', extent=[0, 10, 0, 10])

    # show the 3D rotated projection
    ax4 = fig.add_subplot(224, projection='3d')
    cset = ax4.contourf(xx, yy, volume[:, :, 0], 100, zdir='z', offset=0, cmap=plt.get_cmap('gray'))
    cset = ax4.contourf(xx, yy, volume[:, :, 1], 100, zdir='z', offset=5, cmap=plt.get_cmap('gray'))
    cset = ax4.contourf(xx, yy, volume[:, :, 2], 100, zdir='z', offset=10, cmap=plt.get_cmap('gray'))

    ax4.set_xlim((0., 10.))
    ax4.set_ylim((0., 10.))
    ax4.set_zlim((0., 10.))

    plt.colorbar(cset)
    plt.show()

def get_model_checkpoint(checkpoint_path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max')

    return model_checkpoint_callback

