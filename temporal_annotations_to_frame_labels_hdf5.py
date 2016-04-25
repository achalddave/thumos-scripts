"""Convert temporal annotations (in JSON format) to frame labels."""

import argparse
from math import ceil, floor

import h5py
import numpy as np
from tqdm import tqdm

from util.annotation import (Annotation, annotations_to_frame_labels,
                             filter_annotations_by_category,
                             load_annotations_json)
from util.parsing import load_class_mapping, parse_frame_info_file


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
    parser.add_argument(
        '--sample_frame_rate',
        default=None,
        type=float,
        help="""If specified, the frame labels are output at this frame rate,
                instead of the video's intrinsic frame rate. This allows you to
                dump frame labels at the same frame rate that may have been
                used to dump images.""")
    parser.add_argument('output_labels_hdf5', help='Output HDF5 path')

    args = parser.parse_args()

    annotations = load_annotations_json(args.temporal_annotations_json)
    # Update {start,end}_frame fields to be in terms of the sampled frame rate.
    if args.sample_frame_rate is not None:
        for _, file_annotations in annotations.items():
            for i in range(len(file_annotations)):
                annotation = file_annotations[i]
                new_annotation = annotation._asdict()
                new_annotation['start_frame'] = int(floor(
                    annotation.start_seconds * args.sample_frame_rate))
                new_annotation['end_frame'] = int(ceil(
                    annotation.end_seconds * args.sample_frame_rate))
                file_annotations[i] = Annotation(**new_annotation)

    label_id_to_str = load_class_mapping(args.class_mapping)
    num_labels = len(label_id_to_str)

    fps_num_frames = parse_frame_info_file(args.video_frames_info)
    if args.sample_frame_rate is not None:
        # Compute number of frames under the sampled frame rate.
        num_frames = {}
        for filename, (file_fps, file_num_frames) in fps_num_frames.items():
            num_frames[filename] = resampled_frame_offset(
                file_num_frames, file_fps, args.sample_frame_rate)
    else:
        num_frames = {filename: file_num_frames
                      for filename, (_, file_num_frames) in
                      fps_num_frames.items()}

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
        for filename, file_frame_labels in tqdm(frame_labels.items()):
            output_file[filename] = file_frame_labels


if __name__ == "__main__":
    main()
