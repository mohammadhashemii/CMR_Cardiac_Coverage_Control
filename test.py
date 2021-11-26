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
parser.add_argument('--test_size', type=float, default=0.2, help='test size ratio. this is not important if you use kfold cross validation')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--augmentation', type=bool, default=True, help='whether the training data has been augmented')
parser.add_argument('--transformations', type=list, default=['rotate', 'flip_horizontally', 'flip_vertically', 'brightness'], help='just some transformation supported')
parser.add_argument('--image_size', type=tuple, default=(128, 128), help='image size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--clip_norm', type=float, default=0.35, help='clip norm')
parser.add_argument('--fold_no', type=int, default=0, help='fold number')

args = parser.parse_args()

# transform dataset
augmentation = Augmentation_3D(transformations=args.transformations)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
with tf.device(device_name=device_name):
    # create data loader
    data_loader = DataLoader(hdf5_path=args.data_root + args.dataset)
    dset_x, dset_y = data_loader.read_data()
    train_test_indices = list(kfold.split(dset_x, dset_y))
    train_idx, test_idx = train_test_indices[args.fold_no]
    train_size = len(dset_x[train_idx])
    test_size = len(dset_x[test_idx])
    train_loader = tf.data.Dataset.from_tensor_slices((dset_x[train_idx], dset_y[train_idx]))
    test_loader = tf.data.Dataset.from_tensor_slices((dset_x[test_idx], dset_y[test_idx]))

    del dset_x, dset_y, data_loader
    print("Preparing train/test datasets ...")
    # prepare train dataset
    #           training data
    train_dataset_original = (
        train_loader.shuffle(train_size)
        .map(augmentation.validation_preprocessing)
        .repeat()
        .batch(args.batch_size)
        .prefetch(2)
    )
    if args.augmentation:
        train_dataset_augmented = (
            train_loader.shuffle(train_size)
            .map(augmentation.train_preprocessing)
            .repeat()
            .batch(args.batch_size)
            .prefetch(2)
        )
        train_dataset = train_dataset_original.concatenate(train_dataset_augmented)
        del train_dataset_original, train_dataset_augmented
    else:
        train_dataset = train_dataset_original
        del train_dataset_original

    #           test data
    test_dataset = (
        train_loader.shuffle(test_size)
        .map(augmentation.validation_preprocessing)
        .repeat()
        .batch(args.batch_size)
        .prefetch(2)
    )

    # we don't need these anymore
    #del X_train, X_test, y_train, y_test
    del train_loader, test_loader

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
    train_loss, train_acc = model.evaluate(train_dataset, verbose=1, steps=train_size//args.batch_size)
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1, steps=test_size//args.batch_size)

    print("Train loss: {}".format(train_loss))
    print("Train accuracy: {}".format(train_acc))
    print("Test loss: {}".format(test_loss))
    print("Test accuracy: {}".format(test_acc))



















