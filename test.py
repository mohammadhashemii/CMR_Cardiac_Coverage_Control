import argparse
import os

from data_loader import DataLoader
from augmentation_3D import Augmentation_3D
import tensorflow as tf
from tensorflow.python.client import device_lib
from model import CNN3D
from sklearn.model_selection import KFold

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
device_name = tf.test.gpu_device_name()
print("Available Devices: {}".format(device_lib.list_local_devices()))
print("GPU name: {}".format(tf.test.gpu_device_name()))

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--dataset', type=str, default='dataset_APEX.hdf5', help='dataset name')
parser.add_argument('--model_dir', type=str, default='models/', help='model directory')
parser.add_argument('--weights_path', type=str, help='weights path')
# training
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--augmentation', type=bool, default=True, help='whether the training data has been augmented')
parser.add_argument('--transformations', type=list, default=['rotate', 'flip_horizontally', 'flip_vertically', 'brightness'], help='just some transformation supported')
parser.add_argument('--image_size', type=tuple, default=(128, 128), help='image size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--clip_norm', type=float, default=0.35, help='clip norm')

args = parser.parse_args()

# transform dataset
augmentation = Augmentation_3D(transformations=args.transformations)
with tf.device(device_name=device_name):
    # create data loader
    data_loader = DataLoader(hdf5_path=args.data_root + args.dataset)
    dset_x, dset_y = data_loader.read_data()
    test_size = len(dset_x)
    test_loader = tf.data.Dataset.from_tensor_slices((dset_x, dset_y))

    del dset_x, dset_y, data_loader

    #           test data
    test_dataset = (
        test_loader.shuffle(test_size)
        .map(augmentation.validation_preprocessing)
        .repeat()
        .batch(args.batch_size)
        .prefetch(2)
    )

    del test_loader

    print("Building the model ...")
    # Define model
    cnn_3d = CNN3D(img_size=args.image_size, training=False)
    model = cnn_3d.build()
    loss_fn = tf.keras.losses.binary_crossentropy
    SGD = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                  momentum=args.momentum,
                                  clipnorm=args.clip_norm,
                                  name='SGD')
    model.compile(loss=loss_fn,
                  optimizer=SGD,
                  metrics=['accuracy'])

    model.load_weights(args.weights_path)
    print("Weights loaded from: {}".format(args.weights_path))
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1, steps=test_size//args.batch_size)

    print("Test loss: {}".format(test_loss))
    print("Test accuracy: {}".format(test_acc))



















