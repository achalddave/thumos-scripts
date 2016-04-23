"""Convert temporal annotations (in JSON format) to frame labels."""

import argparse
import h5py
import numpy as np

from util.annotation import (annotations_to_frame_labels,
                             filter_annotations_by_category,
                             load_annotations_json)
from util.parsing import load_class_mapping, parse_frame_info_file


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'temporal_annotations_json',
        help="""Path to temporal annotations JSON file, as output by
                parse_temporal_annotations.py.""")
    parser.add_argument(
        'video_frames_info',
        help='CSV of format <video_name>,<fps>,<num_frames_in_video>')
    parser.add_argument(
        'class_mapping',
        help="""File containing lines of the form "<class_index> <class_name>".
                The order of lines in this file will correspond to the order of
                the labels in the output label matrix.""")
    parser.add_argument('output_labels_hdf5', help='Output HDF5 path')

    args = parser.parse_args()

    annotations = load_annotations_json(args.temporal_annotations_json)
    label_id_to_str = load_class_mapping(args.class_mapping)
    num_labels = len(label_id_to_str)

    # Maps filename to (fps, num_frames); then we change the mapping to be to
    # num_frames only.
    num_frames = parse_frame_info_file(args.video_frames_info)
    num_frames = {filename: fps_num_frames[1]
                  for filename, fps_num_frames in num_frames.items()}

    # Maps filenames to binary matrices of shape (num_frames, num_labels).
    frame_labels = {filename: np.zeros((num_frames[filename], num_labels))
                    for filename in annotations.keys()}

    for i, (label_id, label_str) in enumerate(label_id_to_str.items()):
        label_annotations = filter_annotations_by_category(annotations,
                                                           label_str)
        for filename, file_annotations in label_annotations.items():
            frame_labels[filename][:, i] = annotations_to_frame_labels(
                file_annotations, num_frames[filename])

    with h5py.File(args.output_labels_hdf5, 'w') as output_file:
        for filename, file_frame_labels in frame_labels.items():
            output_file[filename] = file_frame_labels


if __name__ == "__main__":
    main()
