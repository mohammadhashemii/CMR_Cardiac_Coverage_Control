import argparse
import os
from data_loader import DataLoader
from augmentation_3D import Augmentation_3D
import tensorflow as tf
from sklearn.model_selection import KFold
from model import CNN3D
from utils import get_model_checkpoint
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--dataset_name', type=str, default='APEX', help='dataset name. APEX or BASAL')
parser.add_argument('--dataset', type=str, default='dataset_APEX.hdf5', help='dataset file name')
parser.add_argument('--weights_dir', type=str, default='weights/KFold/', help='weights directory')
parser.add_argument('--results_dir', type=str, default='results/', help='results directory for kfold')
# training
parser.add_argument('--no_folds', type=int, default=2, help='number of folds')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--augmentation', type=bool, default=True, help='whether the training data has been augmented')
parser.add_argument('--transformations', type=list, default=['rotate', 'flip_horizontally', 'flip_vertically', 'brightness'], help='just some transformation supported')
parser.add_argument('--image_size', type=tuple, default=(128, 128), help='image size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--clip_norm', type=float, default=0.35, help='clip norm')

args = parser.parse_args()

# make the results and weights directories
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)

# create data loader
data_loader = DataLoader(hdf5_path=args.data_root + args.dataset)
dset_x, dset_y = data_loader.read_data()
# plot a single layer of a random volume
#plot_single_image(dset_x[1,:,:,1])

# transform dataset
augmentation = Augmentation_3D(transformations=args.transformations)

kfold = KFold(n_splits=args.no_folds, shuffle=True)
fold_no = 1
log_file = f"apex_%dfold_results.txt" % args.no_folds
f = open(args.results_dir + log_file, 'w')
f.write("Results for %dfold cross validation. Each fold trained for %d epochs:\n" % (args.no_folds, args.epochs))
f.close()
train_acc_per_fold = []
train_loss_per_fold = []
test_acc_per_fold = []
test_loss_per_fold = []

for train_idx, test_idx in kfold.split(dset_x, dset_y):

    # build the model
    cnn_3d = CNN3D(img_size=args.image_size, training=True)
    model = cnn_3d.build()
    loss_fn = tf.keras.losses.binary_crossentropy
    SGD = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                  momentum=args.momentum,
                                  clipnorm=args.clip_norm,
                                  name='SGD')
    model.compile(loss=loss_fn,
                  optimizer=SGD,
                  metrics=['accuracy'])

    print('-----------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # create model checkpoint callback
    check_point_path = args.weights_dir + args.dataset_name + "_fold_" + str(fold_no) + ".h5"
    model_checkpoint_callback = get_model_checkpoint(checkpoint_path=check_point_path)

    # prepare data
    train_loader = tf.data.Dataset.from_tensor_slices((dset_x[train_idx], dset_y[train_idx]))
    test_loader = tf.data.Dataset.from_tensor_slices((dset_x[test_idx], dset_y[test_idx]))
    train_dataset_original = (
        train_loader.shuffle(len(dset_x[train_idx]))
            .map(augmentation.validation_preprocessing)
            .batch(args.batch_size)
            .prefetch(2)
    )
    if args.augmentation:
        train_dataset_augmented = (
            train_loader.shuffle(len(dset_x[train_idx]))
                .map(augmentation.train_preprocessing)
                .batch(args.batch_size)
                .prefetch(2)
        )
        train_dataset = train_dataset_original.concatenate(train_dataset_augmented)
        del train_dataset_original, train_dataset_augmented
    else:
        train_dataset = train_dataset_original
        del train_dataset_original

    test_dataset = (
        train_loader.shuffle(len(dset_x[test_idx]))
            .map(augmentation.validation_preprocessing)
            .batch(args.batch_size)
            .prefetch(2)
    )

    history = model.fit(train_dataset,
                        validation_data=test_dataset,
                        epochs=args.epochs,
                        callbacks=[model_checkpoint_callback])

    # evaluation
    f = open(args.results_dir + log_file, 'a')
    print('Evaluation for fold {}'.format(fold_no))
    model.load_weights(check_point_path)
    train_loss, train_acc = model.evaluate(train_dataset, verbose=1)
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    f.write("train_accuracy: %.4f, test_accuracy: %.4f, train_loss: %.4f, test_loss: %.4f \n" % (
    train_acc, test_acc, train_loss, test_loss))
    print("train_accuracy: %.4f, test_accuracy: %.4f, train_loss: %.4f, test_loss: %.4f" % (
    train_acc, test_acc, train_loss, test_loss))
    train_acc_per_fold.append(train_acc)
    train_loss_per_fold.append(train_loss)
    test_acc_per_fold.append(test_acc)
    test_loss_per_fold.append(test_loss)
    fold_no = fold_no + 1

print('-----------------------------------------------------')
print("avg_train_acc: %.4f, avg_test_accuracy: %.4f, avg_train_loss: %.4f, avg_test_loss: %.4f" %
      (np.mean(train_acc_per_fold), np.mean(test_acc_per_fold), np.mean(train_loss_per_fold), np.mean(test_loss_per_fold)))
f.write("avg_train_acc: %.4f, avg_test_accuracy: %.4f, avg_train_loss: %.4f, avg_test_loss: %.4f" %
      (np.mean(train_acc_per_fold), np.mean(test_acc_per_fold), np.mean(train_loss_per_fold), np.mean(test_loss_per_fold)))

print('The results has saved in {}.'.format(args.results_dir + log_file))
f.close()