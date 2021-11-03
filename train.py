import argparse
import os
from data_loader import DataLoader
from augmentation_3D import Augmentation_3D
from utils import get_model_checkpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import CNN3D
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--dataset', type=str, default='dataset_APEX.hdf5', help='dataset name')
parser.add_argument('--model_dir', type=str, default='models/', help='model directory')
parser.add_argument('--weights_dir', type=str, default='weights/', help='weights directory')
# training
parser.add_argument('--test_size', type=float, default=0.2, help='test size ratio. this is not important if you use kfold cross validation')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--augmentation', type=bool, default=True, help='whether the training data has been augmented')
parser.add_argument('--transformations', type=list, default=['rotate', 'flip_horizontally', 'flip_vertically', 'brightness'], help='just some transformation supported')
parser.add_argument('--image_size', type=tuple, default=(128, 128), help='image size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--clip_norm', type=float, default=0.35, help='clip norm')

args = parser.parse_args()

# make the model and weights directories
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)



# create data loader
data_loader = DataLoader(hdf5_path=args.data_root + args.dataset)
dset_x, dset_y = data_loader.read_data()
# plot a single layer of a random volume
#plot_single_image(dset_x[1,:,:,1])

# transform dataset
augmentation = Augmentation_3D(transformations=args.transformations)


X_train, X_test, y_train, y_test = train_test_split(dset_x, dset_y, test_size=args.test_size, random_state=42)
train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))

print("Preparing train/test datasets ...")
# prepare train dataset
#           training data
train_dataset_original = (
    train_loader.shuffle(len(X_train))
    .map(augmentation.validation_preprocessing)
    .batch(args.batch_size)
    .prefetch(2)
)
if args.augmentation:
    train_dataset_augmented = (
        train_loader.shuffle(len(X_train))
        .map(augmentation.train_preprocessing)
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
    train_loader.shuffle(len(X_test))
    .map(augmentation.validation_preprocessing)
    .batch(args.batch_size)
    .prefetch(2)
)

# we don't need these anymore
del X_train, X_test, y_train, y_test
del dset_x, dset_y

print("Building the model ...")
# Define model
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

with open(args.model_dir + 'model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
print("Model built. Model summary saved in " + args.model_dir + 'model_summary.txt')
print("=======================")

model_checkpoint_callback = get_model_checkpoint(checkpoint_path=args.weights_dir + 'apex_weights.h5')
# training
history = model.fit(train_dataset,
                    validation_data=test_dataset,
                    epochs=args.epochs,
                    callbacks=[model_checkpoint_callback])

# save the model for later use
tf.keras.models.save_model(model, args.model_dir + 'apex_model.h5')















