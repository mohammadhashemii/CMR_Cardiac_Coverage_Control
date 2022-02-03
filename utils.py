import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import h5py


def plot_single_image(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

def plot_volume(volume, save_fig=False, filename='figure_1.png'):
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
    if save_fig:
        if not os.path.exists('images/'):
            os.makedirs('images/')
        plt.savefig('images/' + filename)
    else:
        plt.show()

    plt.close(fig)

def get_model_checkpoint(checkpoint_path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max')

    return model_checkpoint_callback

def merge_hdf5_files(file_paths: list, merged_file_path: str):

    idx = np.array([])
    mask = np.array([])
    #X = np.array([])
    Y = np.array([])
    for fp in file_paths:
        with h5py.File(fp, 'r') as hf:
            idx = np.concatenate([idx, np.array(hf['idx'])]) if idx.size else np.array(hf['idx'])
            mask = np.concatenate([mask, np.array(hf['mask'])]) if mask.size else np.array(hf['mask'])
            #X = np.concatenate([X, np.array(hf['X'])]) if X.size else np.array(hf['X'])
            Y = np.concatenate([Y, np.array(hf['Y'])]) if Y.size else np.array(hf['Y'])

    with h5py.File(merged_file_path, 'w') as hf:
        hf.create_dataset('idx', data=idx, shape=idx.shape, compression='gzip', chunks=True)
        hf.create_dataset('mask', data=mask, shape=mask.shape, compression='gzip', chunks=True)
        #hf.create_dataset('X', data=X, shape=X.shape, compression='gzip', chunks=True)
        hf.create_dataset('Y', data=Y, shape=Y.shape, compression='gzip', chunks=True)

    print(mask.shape, Y.shape, idx.shape)
file_paths= ['data/apex_lime/masks_apex_1.hdf5',
             'data/apex_lime/masks_apex_2.hdf5']
merge_hdf5_files(file_paths=file_paths,
                 merged_file_path='data/apex_lime/masks_apex.hdf5')

