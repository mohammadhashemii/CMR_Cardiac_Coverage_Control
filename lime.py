import os
import argparse
import copy
import h5py
from data_loader import DataLoader
from augmentation_3D import Augmentation_3D
import skimage.segmentation
import numpy as np
from utils import plot_volume
from sklearn.model_selection import train_test_split
import tensorflow as tf

parser = argparse.ArgumentParser()

# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--dataset', type=str, default='dataset_APEX.hdf5', help='dataset name')
parser.add_argument('--model_path', type=str, default='models/apex_model.h5', help='path to the model')
parser.add_argument('--weights_path', type=str, default='weights/apex_weights.h5', help='path to the model weights for testing')
parser.add_argument('--best_perturbations_dir', type=str, default='perturbations/', help='path to the perturbations directory')
parser.add_argument('--iterations', type=int, default=500, help='number of times which we want to apply different random perturbations')
parser.add_argument('--transformations', type=list, default=['rotate', 'flip_horizontally', 'flip_vertically', 'brightness'], help='just some transformation supported')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')

args = parser.parse_args()

# create data loader
data_loader = DataLoader(hdf5_path=args.data_root + args.dataset)
dset_x, dset_y = data_loader.read_data()

if not os.path.exists(args.best_perturbations_dir):
    os.makedirs(args.best_perturbations_dir)

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
        print("Generating perturbations ...")
        assert superpixels.shape == self.volume.shape

        layers_perturbation = []
        for i in range(superpixels.shape[-1]): # over volume layers
            n_unique_values = len(np.unique(superpixels[:, :, i]))
            p = np.random.binomial(1, 0.5, size=(1,n_unique_values)).squeeze()
            layers_perturbation.append(p)

        return layers_perturbation

    def apply_perturbations(self, layers_perturbation: list, superpixels):
        print("Applying perturbations on dataset ...")

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
    .batch(args.batch_size)
    .prefetch(2)
)
# we don't need these anymore
del dset_x, dset_y, data

#print(model.predict(dataset))
model.evaluate(dataset)

#volume = dset_x[0]
#lime = Lime(volume)
#superpixels = lime.generate_segmentation(max_iter=1000)
#layers_perturbation = lime.generate_perturbations(superpixels)
#perturbed_volume = lime.apply_perturbations(layers_perturbation, superpixels)
#plot_volume(perturbed_volume)

'''
best_accuracy = 0.0
best_perturbations = None
idx = 0
for i in range(args.iterations):
    print('Iter %d/%d:' % (i, args.iterations))
    print("================================")
    segmentations = lime.generate_segmentation(max_iter=200)
    volume_perturbations = lime.generate_perturbations(segmentations)
    perturbed_volumes = lime.apply_perturbations(volume_perturbations, segmentations)

    print("Evaluating the model ...")
    loss, accuracy = model.evaluate(dset_x)
    if best_accuracy <= accuracy:
        best_accuracy = accuracy
        best_perturbations = volume_perturbations
        idx = i

print("Best accuracy: {}".format(best_accuracy))
perturbations_path = args.best_perturbations_dir + 'perturbations_' + str(idx) + '.h5'
print("Best perturbations saved at: {}".format(perturbations_path))

with h5py.File(perturbations_path, 'w', libver='latest') as hf:
    print(len(best_perturbations))
    for i in range(len(best_perturbations)):
        data = best_perturbations[i]
        #print(data)
        hf.create_dataset(str(idx), data=data, compression='gzip', shape=data.shape, chunks=True)


'''

