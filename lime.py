import argparse
import copy
import h5py
from data_loader import DataLoader
from augmentation_3D import Augmentation_3D
import skimage.segmentation
import numpy as np
from utils import plot_volume
import tensorflow as tf
from tensorflow.python.client import device_lib

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
device_name = tf.test.gpu_device_name()
print("Available Devices: {}".format(device_lib.list_local_devices()))
print("GPU name: {}".format(tf.test.gpu_device_name()))

parser = argparse.ArgumentParser()

# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--dataset', type=str, default='dataset_APEX.hdf5', help='dataset name')
parser.add_argument('--model_path', type=str, default='models/apex_model.h5', help='path to the model')
parser.add_argument('--weights_path', type=str, default='weights/_fold0_apex_weights.h5', help='path to the model weights for testing')

parser.add_argument('--iterations', type=int, default=1000, help='number of times which we want to apply different random perturbations')
parser.add_argument('--transformations', type=list, default=['rotate', 'flip_horizontally', 'flip_vertically', 'brightness'], help='just some transformation supported')
parser.add_argument('--n_pert', type=int, default=10, help='number of random generated perturbations for each sample')
args = parser.parse_args()

# create data loader
data_loader = DataLoader(hdf5_path=args.data_root + args.dataset)
dset_x, dset_y = data_loader.read_data()

class Lime():
    def __init__(self, volume):
        self.volume = volume

    def generate_segmentation(self, n_segments=70, compactness=0.3, max_iter=1000):
        '''
        generate 3 segmentation for images of a volume
        '''
        superpixels = []
        for i in range(self.volume.shape[-1]):   # over volume layers
            layer = self.volume[:, :, i]  # layer is a 2D array now
            temp_volume = np.repeat(layer[None, :], 3, axis=0).transpose(1,2,0)  # create a 3-layer tensor by repeating the layer
            super_pixel = skimage.segmentation.slic(temp_volume,
                                                    n_segments=n_segments,
                                                    compactness=compactness,
                                                    max_iter=max_iter,
                                                    start_label=1)
            superpixels.append(super_pixel)
        superpixels = np.array(superpixels).transpose(1, 2, 0)

        return superpixels

    def generate_perturbations(self, superpixels) -> list:
        assert superpixels.shape == self.volume.shape

        layers_perturbation = []
        for i in range(superpixels.shape[-1]): # over volume layers
            n_unique_values = len(np.unique(superpixels[:, :, i]))
            p = np.random.binomial(1, 0.5, size=(1,n_unique_values)).squeeze()
            layers_perturbation.append(p)

        return layers_perturbation

    def apply_perturbations(self, layers_perturbation: list, superpixels):

        perturbed_volume = []

        for i in range(len(layers_perturbation)):  # loop over the layers of a volume
            active_pixels = np.where(layers_perturbation[i] == 1)[0]
            mask = np.zeros(superpixels[:, :, i].shape)
            for active in active_pixels:
                mask[superpixels[:, :, i] == active] = 1

            perturbed_image = copy.deepcopy(self.volume[:, :, i])
            perturbed_image = perturbed_image * mask
            perturbed_volume.append(perturbed_image)

        perturbed_volume = np.array(perturbed_volume).transpose(1, 2, 0)

        return perturbed_volume


# load trained model for evaluating
print("Model loaded from: {}".format(args.model_path))
model = tf.keras.models.load_model(args.model_path)
print("Weights loaded from: {}".format(args.weights_path))
model.load_weights(args.weights_path)

# transform dataset
augmentation = Augmentation_3D(transformations=args.transformations)
data = tf.data.Dataset.from_tensor_slices((dset_x, dset_y))

print("Preparing datasets ...")
dataset = (
    data.shuffle(len(data))
    .map(augmentation.validation_preprocessing)
    .batch(1)       # since we want to have prediction for each sample
    .prefetch(2)
)
data_size = len(data)
# we don't need these anymore
del dset_x, dset_y, data

correct_predicted_samples_X = []
correct_predicted_samples_Y = []
best_perturbations_X = []
best_perturbations_Y = []
idx = 0
with tf.device(device_name=device_name):
    for sample in dataset:
        print("============================")
        print('Processing on sample:{}/{}'.format(idx, data_size))
        volume = sample[0][0, :, :, :, 0]  # volume shape must be (128, 128, 3)
        target = sample[1].numpy()[0, 0]
        temp_volume = tf.expand_dims(volume, axis=3)
        temp_volume = tf.expand_dims(temp_volume, axis=0)
        pred = model.predict(temp_volume)[0,0]
        if (pred >= 0.5 and target == 0) or (pred < 0.5 and target == 1): # wrong prediction and pass it
            print("IGNORED. The prediction was wrong.")
            idx += 1
            continue
        else:
            correct_predicted_samples_X.append(volume)
            correct_predicted_samples_Y.append(target)
        lime = Lime(volume)
        best_pred = 0.5
        best_volume = None  # generated volume by lime with the most accurate prediction
        best_idx = 0
        for i in range(args.n_pert):
            superpixels = lime.generate_segmentation(max_iter=args.iterations)
            layers_perturbation = lime.generate_perturbations(superpixels)
            perturbed_volume = lime.apply_perturbations(layers_perturbation, superpixels)
            #plot_volume(perturbed_volume)
            temp_volume = tf.expand_dims(perturbed_volume, axis=3)
            temp_volume = tf.expand_dims(temp_volume, axis=0)
            pred = model.predict(temp_volume)[0,0]
            if target == 1:
                if  pred > best_pred:
                    best_pred = pred
                    best_volume = volume
                    best_idx = i
            else:   # target = 0
                if  pred < best_pred:
                    best_pred = pred
                    best_volume = volume
                    best_idx = i
        print("The {}/{} perturbation is chosen with prediction score: {}".format(best_idx, args.n_pert, best_pred))
        best_perturbations_X.append(best_volume)
        best_perturbations_Y.append(target)
        idx += 1


correct_predicted_samples_X = np.array(correct_predicted_samples_X)
correct_predicted_samples_Y = np.array(correct_predicted_samples_Y)
best_perturbations_X = np.array(best_perturbations_X)
best_perturbations_Y = np.array(best_perturbations_Y)

with h5py.File(args.data_root + 'correct_predictions.hdf5', 'w') as hf:
    hf.create_dataset('X', data=correct_predicted_samples_X, shape=correct_predicted_samples_X.shape, compression='gzip', chunks=True)
    hf.create_dataset('Y', data=correct_predicted_samples_Y, shape=(len(correct_predicted_samples_Y), 1), compression='gzip', chunks=True)

with h5py.File(args.data_root + 'perturbations.hdf5', 'w') as hf:
    hf.create_dataset('X', data=best_perturbations_X, shape=best_perturbations_X.shape, compression='gzip', chunks=True)
    hf.create_dataset('Y', data=best_perturbations_Y, shape=(len(best_perturbations_Y), 1), compression='gzip', chunks=True)



