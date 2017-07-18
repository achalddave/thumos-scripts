"""Convert temporal annotations (in JSON format) to frame labels."""

import argparse
import collections
import glob
import logging
from itertools import chain

import h5py
import numpy as np
from tqdm import tqdm

from util.video_tools.util.annotation import (collect_frame_labels,
                                              load_label_ids)
from util.video_tools.frame_loader_util import parse_frame_path
from util.parsing import (load_thumos_annotations)


def resampled_frame_offset(frame_offset, original_fps, sampled_fps):
    """Compute the frame offset for a given frame if the video was resampled.

    >>> resampled_frame_offset(3, 10, 1)
    0
    >>> resampled_frame_offset(3, 5, 1)
    0
    >>> resampled_frame_offset(3, 3, 1)
    1
    """
    return int(round(frame_offset * sampled_fps / original_fps))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    parser._action_groups.append(optional)
    required.add_argument(
        '--annotations',
        required=True,
        help="""Annotations directory as distributed by MultiTHUMOS.""")
    required.add_argument(
        '--frames_root',
        required=True,
        help="""Root directory containing train/, val/, test/ directories. Each
                subdirectory should contain <video_name>/frame%%04d.png,
                starting with frame0.png""")
    required.add_argument(
        '--video_frames_info',
        required=True,
        help='CSV of format <video_name>,<fps>,<num_frames_in_video>')
    required.add_argument(
        '--class_mapping',
        required=True,
        help="""File containing lines of the form "<class_index> <class_name>".
                The order of lines in this file will correspond to the order of
                the labels in the output label matrix.""")
    optional.add_argument(
        '--sample_frame_rate',
        default=10,
        type=float,
        help="""If specified, the frame labels are output at this frame rate,
                instead of the video's intrinsic frame rate. This allows you to
                dump frame labels at the same frame rate that may have been
                used to dump images.""")

    required.add_argument(
        '--output_trainval_hdf5', help='Output HDF5 path', required=True)
    required.add_argument(
        '--output_test_hdf5', help='Output HDF5 path', required=True)

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')

    annotations = load_thumos_annotations(args.annotations,
                                          args.video_frames_info)
    file_annotations = collections.defaultdict(list)
    for annotation in annotations:
        file_annotations[annotation.filename].append(annotation)

    logging.info('Collecting train paths.')
    train_paths = list(
        tqdm(
            glob.iglob('{}/train_temporal/*/*.png'.format(args.frames_root)),
            leave=False))
    logging.info('Collected %d paths.', len(train_paths))
    logging.info('Collecting validation paths.')
    val_paths = list(
        tqdm(
            glob.iglob('{}/validation_temporal/*/*.png'.format(
                args.frames_root)),
            leave=False))
    logging.info('Collected %d paths.', len(val_paths))
    logging.info('Collecting test paths.')
    test_paths = list(
        tqdm(
            glob.iglob('{}/test_temporal/*/*.png'.format(args.frames_root)),
            leave=False))
    logging.info('Collected %d paths.', len(test_paths))

    label_ids = load_label_ids(args.class_mapping, one_indexed_labels=True)
    num_labels = len(label_ids)

    num_frames = collections.Counter()

    for frame_path in chain(train_paths, val_paths, test_paths):
        video_name, _ = parse_frame_path(frame_path)
        num_frames[video_name] += 1

    trainval_labels = {}
    logging.info('Processing training frames.')
    for frame_path in tqdm(train_paths):
        video_name, frame_number = parse_frame_path(frame_path)
        frame_number -= 1
        if video_name not in trainval_labels:
            trainval_labels[video_name] = np.zeros(
                (num_frames[video_name], num_labels))

        # Videos are of the form 'v_<label>_g<number>_c<number>'
        frame_label = video_name.split('_')[1]
        frame_label_id = label_ids[frame_label]
        trainval_labels[video_name][frame_number, frame_label_id] = 1

    logging.info('Processing validation frames.')
    for frame_path in tqdm(val_paths):
        video_name, frame_number = parse_frame_path(frame_path)
        frame_number -= 1
        if video_name not in trainval_labels:
            trainval_labels[video_name] = np.zeros(
                (num_frames[video_name], num_labels))

        frame_labels = collect_frame_labels(
            file_annotations[video_name],
            frame_number,
            frames_per_second=args.sample_frame_rate)
        frame_label_ids = [label_ids[label] for label in frame_labels]
        trainval_labels[video_name][frame_number, frame_label_ids] = 1

    test_labels = {}
    logging.info('Processing test frames.')
    for frame_path in tqdm(test_paths):
        video_name, frame_number = parse_frame_path(frame_path)
        frame_number -= 1
        if video_name not in test_labels:
            test_labels[video_name] = np.zeros(
                (num_frames[video_name], num_labels))

        frame_labels = collect_frame_labels(
            file_annotations[video_name],
            frame_number,
            frames_per_second=args.sample_frame_rate)
        frame_label_ids = [label_ids[label] for label in frame_labels]
        test_labels[video_name][frame_number, frame_label_ids] = 1

    with h5py.File(args.output_trainval_hdf5, 'w') as output_file:
        for filename, file_frame_labels in tqdm(trainval_labels.items()):
            output_file[filename] = file_frame_labels

    with h5py.File(args.output_test_hdf5, 'w') as output_file:
        for filename, file_frame_labels in tqdm(test_labels.items()):
            output_file[filename] = file_frame_labels


if __name__ == "__main__":
    main()
