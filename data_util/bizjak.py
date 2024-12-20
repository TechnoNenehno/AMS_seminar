"""
Utitlities for Bizjak datasets
Modified from
https://github.com/microsoft/Recursive-Cascaded-Networks/tree/master/data_util/liver.py
"""


import numpy as np
import json
import os
import h5py
import _pickle as pickle
from .data import Split
from scipy.ndimage import zoom
import nibabel as nib

def get_range(imgs):
    r = np.any(imgs.reshape(imgs.shape[0], -1), axis=-1).nonzero()
    return np.min(r), np.max(r)

class Split:
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

class Hdf5Reader:
    def __init__(self, path):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('{} not found!'.format(path))
            self.file = None

    def __getitem__(self, key):
        data = {'id': key}
        if self.file is None:
            return data
        group = self.file[key]
        for k in group:
            data[k] = group[k]
        return data


class FileManager:
    def __init__(self, files):
        self.files = {}
        for k, v in files.items():
            self.files[k] = Hdf5Reader(v["path"])

    def __getitem__(self, key):
        p = key.find('/')
        if key[:p] in self.files:
            ret = self.files[key[:p]][key[p+1:]]
            ret['id'] = key.replace('/', '_')
            return ret
        elif '/' in self.files:
            ret = self.files['/'][key]
            ret['id'] = key.replace('/', '_')
            return ret
        else:
            raise KeyError('{} not found'.format(key))




class Dataset:
    def __init__(self, args, split_path, paired=True, batch_size=None):
        # Load JSON config
        with open(split_path, 'r') as f:
            config = json.load(f)
        
        self.files = FileManager(config.get('files', {}))
        self.subset = {}
        self.paired = paired
        self.batch_size = batch_size

        # Extract target tensor shape from JSON
        tensor_shape = config.get("tensorImageShape", {}).get("0", None)
        if tensor_shape is None:
            raise ValueError("`tensorImageShape` is missing in the JSON file.")
        self.image_size = (128,128,128)# (256, 192, 192) in this case

        # Extract training pairs
        self.training_paired_images = config.get("training_paired_images", [])
        if not self.training_paired_images:
            raise ValueError("No training pairs defined in 'training_paired_images'.")
        
        # Load validation pairs
        self.validation_paired_images = config.get("registration_val", [])
        if not self.validation_paired_images:
            print("Warning: No validation pairs defined in 'registration_val'.")

    def generate_pairs(self, subset='train', loop=False):
        """
        Yield fixed-moving pairs from the specified subset.

        Args:
            subset (str): 'train' for training pairs or 'valid' for validation pairs.
            loop (bool): Whether to loop through the dataset endlessly.

        Yields:
            Tuple[Dict, Dict]: Fixed and moving image dictionaries.
        """
        if subset == 'train':
            paired_images = self.training_paired_images
        elif subset == 'valid':
            paired_images = self.validation_paired_images
        else:
            raise ValueError(f"Unknown subset: {subset}")

        while True:
            if loop:
                np.random.shuffle(paired_images)
            for pair in paired_images:
                fixed_path = pair['fixed']
                moving_path = pair['moving']
                fixed = self.load_image(fixed_path)  # Load fixed image
                moving = self.load_image(moving_path)  # Load moving image
                yield fixed, moving
            if not loop:
                break

    def load_image(self, path):
        """
        Load a 3D image from a .nii.gz file and preprocess it to the desired size.

        Args:
            path (str): Path to the NIfTI (.nii.gz) file.

        Returns:
            dict: A dictionary containing the preprocessed image volume and its ID.
        """
        try:
            nii = nib.load(path)  # Load the NIfTI file
            image = nii.get_fdata()  # Extract the image data as a NumPy array
            image = self.preprocess_image(image)  # Resize and normalize the image
            return {"volume": image, "id": os.path.basename(path)}
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            raise

    def resize_image(self, image, target_size=(128, 128, 128)):
            """
            Resize a 3D image to the target size using linear interpolation.

            Args:
                image (numpy.ndarray): Input 3D image to be resized.
                target_size (tuple): Desired output size (depth, height, width).

            Returns:
                numpy.ndarray: Resized 3D image.
            """
            factors = [t / s for t, s in zip(target_size, image.shape)]
            resized_image = zoom(image, factors, order=2)  # Order=1 for linear interpolation
            return resized_image
            
    def preprocess_image(self, image):
        """
        Preprocess a 3D image: resize and normalize.

        Args:
            image (numpy.ndarray): The raw 3D image data.

        Returns:
            numpy.ndarray: The preprocessed image.
        """
        # Resize the image to 128x128x128
        image = self.resize_image(image, target_size=(128, 128, 128))

        self.image_size = (128, 128, 128)
        # Normalize the image to [0, 1] range
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:  # Avoid division by zero
            image = (image - min_val) / (max_val - min_val)
        
        return image

    def generator(self, subset='train', batch_size=None, loop=False):
        """
        Generate batches of data for the specified subset.

        Args:
            subset (str): 'train' or 'valid'.
            batch_size (int): Number of image pairs per batch.
            loop (bool): Whether to loop through the dataset endlessly.

        Yields:
            Dict: Batch of fixed and moving image pairs, with metadata.
        """
        if batch_size is None:
            batch_size = self.batch_size
        pairs_gen = self.generate_pairs(subset=subset, loop=loop)

        while True:
            ret = dict()
            ret['voxel1'] = np.zeros(
                (batch_size, *self.image_size, 1), dtype=np.float32
            )
            ret['voxel2'] = np.zeros(
                (batch_size, *self.image_size, 1), dtype=np.float32
            )
            ret['seg1'] = np.ones(
                (batch_size, *self.image_size, 1), dtype=np.float32
            )
            ret['seg2'] = np.ones(
                (batch_size, *self.image_size, 1), dtype=np.float32
            )
            ret['point1'] = np.empty(
                (batch_size, 6,3), dtype=np.float32
            )
            ret['point2'] = np.empty(
                (batch_size, 6,3), dtype=np.float32
            )
            ret['id1'] = np.empty((batch_size), dtype='<U40')
            ret['id2'] = np.empty((batch_size), dtype='<U40')

            for i in range(batch_size):
                try:
                    fixed, moving = next(pairs_gen)
                except StopIteration:
                    return  # End of data

                ret['voxel1'][i, ..., 0] = fixed['volume']
                ret['voxel2'][i, ..., 0] = moving['volume']
                ret['id1'][i] = fixed['id']
                ret['id2'][i] = moving['id']
            yield ret
