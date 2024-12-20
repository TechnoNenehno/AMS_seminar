"""
For evaluate the network
Modified from
https://github.com/microsoft/Recursive-Cascaded-Networks/blob/master/eval.py

"""

import argparse
import os
import json
import re
import time
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch', type=int, default=4, help='Size of minibatch')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--paired', action='store_true')
parser.add_argument('--data_args', type=str, default=None)
parser.add_argument('--net_args', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--saved_file', type=str)
parser.add_argument('--flow_multiplier', type=float, default=1.)
parser.add_argument('--eval_output_dir', type=str, default='evaluate')
parser.add_argument('--writing_flow', action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn
tf.logging.set_verbosity(tf.logging.ERROR)

import network
import data_util.bizjak
from data_util.bizjak import Split


import numpy as np
from scipy.ndimage import zoom

import numpy as np
from scipy.ndimage import zoom

def resize_image(image, target_size=(256, 192, 192)):
    """
    Resize a 4D image to the target size for the first three axes (depth, height, width),
    leaving the fourth axis (channels) unchanged.

    Args:
        image (numpy.ndarray): Input 4D image to be resized, with shape (depth, height, width, channels).
        target_size (tuple): Desired output size (depth, height, width) for the first three axes.

    Returns:
        numpy.ndarray: Resized 4D image with the same number of channels.
    """
    # Ensure the input image is 4D
    if len(image.shape) != 4:
        raise ValueError("Input image must be 4D.")

    # Ensure the target size is 3-dimensional (depth, height, width)
    if len(target_size) != 3:
        raise ValueError("Target size must be a 3D tuple (depth, height, width).")

    # Get the original shape
    original_shape = image.shape
    
    # Check that the first three dimensions are in the target size
    original_depth, original_height, original_width, channels = original_shape
    target_depth, target_height, target_width = target_size
    
    # Calculate the scaling factors for each of the first three axes
    factors = [target_depth / original_depth, target_height / original_height, target_width / original_width]
    
    # Resize the image only on the first three axes, leaving the 4th axis (channels) unchanged
    resized_image = zoom(image, zoom=factors + [1], order=2)  # Apply zoom on the first three axes

    return resized_image



def create_new_name(fname):
    """
    Splits the file name at underscores (_), selects parts at indexes 3, 4, 6, and 7,
    removes ".nii.gz" if present, and combines them into a new name.

    Args:
        fname (str): The original file name.

    Returns:
        str: The new name formed by combining cleaned parts at indexes 3, 4, 6, and 7.
    """
    # Split the name at underscores
    parts = fname.split("_")
    
    # Ensure the file name has enough parts
    required_indexes = [3, 4, 6, 7]
    if len(parts) <= max(required_indexes):
        raise ValueError("The file name does not have enough parts after splitting.")
    
    # Extract and clean parts at the required indexes
    selected_parts = [
        parts[i].replace(".nii.gz", "") for i in required_indexes
    ]
    # Insert an underscore every 4th index
    result = ""
    for i, part in enumerate(selected_parts):
        result += part
        if(i == len(selected_parts)-1):
            break
        result += "_"
    
    # Combine the selected parts into a new name with "disp_" in front
    return "disp_" + result

def main():
    if args.checkpoint is None:
        print('Checkpoint must be specified!')
        return
    if ':' in args.checkpoint:
        args.checkpoint, steps = args.checkpoint.split(':')
        steps = int(steps)
    else:
        steps = None
    args.checkpoint = find_checkpoint_step(args.checkpoint, steps)
    print(args.checkpoint)
    model_dir = os.path.dirname(args.checkpoint)
    try:
        with open(os.path.join(model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        print(model_args)
    except Exception as e:
        print(e)
        model_args = {}

    if args.dataset is None:
        args.dataset = model_args['dataset']
    if args.data_args is None:
        args.data_args = model_args['data_args']

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = model_args['base_network']
    Framework.net_args['n_cascades'] = model_args['n_cascades']
    Framework.net_args['dataset'] = model_args['dataset']
    Framework.net_args['flow_multiplier'] = args.flow_multiplier
    Framework.net_args['rep'] = args.rep
    Framework.net_args.update(eval('dict({})'.format(model_args['net_args'])))
    if args.net_args is not None:
        Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = "128,128,128"
        image_type = "bizjak"
    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))
    framework = Framework(devices=gpus, image_size=args.image_size,
                            segmentation_class_value=cfg.get(
                                'segmentation_class_value', None),
                            fast_reconstruction=args.fast_reconstruction,
                            validation=True)
    print('Graph built')

    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    ds = Dataset(args, args.dataset,
                    batch_size=args.batch, paired=args.paired,
                    **eval('dict({})'.format(args.data_args)))

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    tf.global_variables_initializer().run(session=sess)

    saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES))
    checkpoint = args.checkpoint
    saver.restore(sess, checkpoint)
    tflearn.is_training(False, session=sess)

    #load validation subsets
    val_subsets = [Split.VALID]
    if args.writing_flow:
        val_subsets = [Split.TRAIN]

    if args.val_subset is not None:
        val_subsets = args.val_subset.split(',')

    tflearn.is_training(False, session=sess)
    keys = ['pt_mask', 'landmark_dists', 'jaccs', 'dices', 'jacobian_det', 'real_flow']
    if not os.path.exists(args.eval_output_dir):
        os.mkdir(args.eval_output_dir)
    path_prefix = os.path.join(args.eval_output_dir,
                        short_name(checkpoint) + "-" + str(args.flow_multiplier))
    if args.rep > 1:
        path_prefix = path_prefix + '-rep' + str(args.rep)
    if args.name is not None:
        path_prefix = path_prefix + '-' + args.name

    for val_subset in val_subsets:
        if args.val_subset is not None:
            output_fname = path_prefix + '-' + str(val_subset) + '.txt'
        else:
            output_fname = path_prefix + '.txt'
        with open(output_fname, 'w') as fo:
            print("Validation subset {}".format(val_subset))
            gen = ds.generator(val_subset, batch_size=args.batch, loop=False)
            results = framework.validate(sess, gen, args, keys=keys,
                    summary=False, show_tqdm=True)
            #shranjevanje deformacijskih polj
            print("cigomiogo")
            name1 = []
            name2 = []
            for name in results['id1']:
                name1.append(name)
            for name in results['id2']:
                name2.append(name)
            for i, array in enumerate(results['real_flow']):
                #interpoliraj slike 
                image = resize_image(array, target_size=(256, 192, 192))
                #split and extract names
                a = (f"deformacijsko_polje_{name1[i]}_{name2[i]}_{i}.npz")
                b = create_new_name(a)
                # sitk.WriteImage(a, f"deformacijsko_polje_{name1[i]}_{name2[i]}_{i}.nii.gz")
                np.savez("/app/outputs/" + b, image)
            #print(results['real_flow'])
            for i in range(len(results['jaccs'])):
                print(results['id1'][i], results['id2'][i],
                        np.mean(results['dices'][i]), np.mean(results['jaccs'][i]),
                        np.mean(results['landmark_dists'][i]),
                        results['jacobian_det'][i], file=fo)
            print('Summary', file=fo)
            jaccs, dices, landmarks = results['jaccs'],\
                                        results['dices'],\
                                        results['landmark_dists']
            jacobian_det = results['jacobian_det']
            print("Dice score: {} ({})".format(np.mean(dices), np.std(
                np.mean(dices, axis=-1))), file=fo)
            print("Jacc score: {} ({})".format(np.mean(jaccs), np.std(
                np.mean(jaccs, axis=-1))), file=fo)
            print("Landmark distance: {} ({})".format(np.mean(landmarks), np.std(
                np.mean(landmarks, axis=-1))), file=fo)
            print("Jacobian determinant: {} ({})".format(np.mean(
                jacobian_det), np.std(jacobian_det)), file=fo)


def short_name(checkpoint):
    cpath, steps = os.path.split(checkpoint)
    _, exp = os.path.split(cpath)
    return exp + '-' + steps


def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps),
                os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]


if __name__ == '__main__':
    main()
