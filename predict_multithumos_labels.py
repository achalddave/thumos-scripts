"""Calculate Multi-THUMOS activity predictions from FC7 features using Caffe.

Using (1) a model trained to predict MultiTHUMOS and (2) FC7 features output
from the model on a set of images, predicts Multi-THUMOS actions for each FC7
feature input using the MultiTHUMOS model.
"""

import argparse
import logging

import caffe
import h5py

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                    datefmt='%H:%M:%S')

ORDERED_CROPS = ['left_flip', 'center_flip', 'right_flip', 'left', 'center',
                 'right']
CROP_INDICES = {crop: str(crop_idx)
                for (crop_idx, crop) in enumerate(ORDERED_CROPS)}

MODEL_CAFFEMODEL = '/scratch/olga/iccv15_feats/fga_spatial_vgg_ft_iter_5000.caffemodel'
MODEL_PROTOTXT = './vgg_ft_deploy_fc7_to_labels.prototxt'

FC7_FEATURE_DIM = 4096


def predict_action_probabilities(net, fc7_features):
    """
    Args:
        net (caffe.Net)
        fc7_features ((num_datapoints, feature_dimension) array)

    Returns:
        predictions ((num_datapoints, num_classes) array): Predictions for each
            input feature.
    """
    input_layer = net.inputs[0]  # FC7
    feature_dim = fc7_features.shape[1]
    if feature_dim != FC7_FEATURE_DIM:
        raise ValueError("Input FC7 features should have {} channels, "
                         "but received %s channels.".format(FC7_FEATURE_DIM,
                                                            feature_dim))
    # Set batch size
    net.blobs[input_layer].reshape(fc7_features.shape[0], FC7_FEATURE_DIM)
    return net.forward_all(**{input_layer: fc7_features})['prob']


def main():
    net = caffe.Net(MODEL_PROTOTXT, MODEL_CAFFEMODEL, caffe.TEST)
    with h5py.File(args.fc7_features, 'r') as features_file, h5py.File(
            args.output_hdf5, 'w') as output_file:
        for crop_index in CROP_INDICES.values():
            logging.info("Calculating predictions for crop %s",
                         ORDERED_CROPS[int(crop_index)])
            output_file.create_group(crop_index)
            for filename, features in features_file[crop_index].items():
                predictions = predict_action_probabilities(net, features)
                output_file[crop_index][filename] = predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fc7_features',
                        help="""
                            HDF5 file containing features as input.
                            features_hdf5[crop][vidName] should be a
                            (num_vid_frames, num_dimensions) array of features.
                            [crop] should range from 0-5, corresponding to crops
                            {crops}""".format(crops=ORDERED_CROPS))
    parser.add_argument('output_hdf5')

    args = parser.parse_args()

    main()
