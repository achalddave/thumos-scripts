"""Parse temporal annotations for THUMOS '14 validation data, output JSON.

Output format:
    [
        {
            filename: ...,
            start_seconds: ...,
            end_seconds: ...,
            start_frame: ...,
            end_frame: ...,
            frames_per_second: ...,
            category: ...
        },
        ...
    ]
"""

import argparse
import csv
import json
import os
from math import ceil, floor
from os import path


def parse_frame_info(video_frames_info_path):
    video_fps = dict()
    with open(video_frames_info_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip headers
        for row in reader:
            video_fps[row[0]] = float(row[1])
    return video_fps


def parse_annotation_file(annotation_path, video_fps):
    annotations = []
    with open(annotation_path) as f:
        for line in f:
            # Format: "<video_name> <start_time> <end_time>" or
            # "<video_name>  <start_time> <end_time>".
            # The THUMOS temporal labels have *two spaces* between the first two
            # fields (unfortunately), while the MultiTHUMOS labels have one
            # space.
            details = line.strip().split(' ')
            if details[1] == '':  # There were two spaces after the first field.
                details.pop(1)
            filename, start, end = details
            start, end = float(start), float(end)
            current_fps = video_fps[filename]
            start_frame = floor(start * current_fps)
            end_frame = ceil(end * current_fps)
            annotations.append({'filename': filename,
                                'start_seconds': start,
                                'end_seconds': end,
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'frames_per_second': current_fps
                                })
    return annotations


def main():
    annotation_paths = ["%s/%s" % (args.input_annotation_dir, x)
                        for x in os.listdir(args.input_annotation_dir)]
    # Maps video name to frames per second
    video_fps = parse_frame_info(args.video_frames_info)
    annotations = []
    for annotation_path in annotation_paths:
        if annotation_path.endswith('_val.txt'):
            # annotation_path is of the form /path/to/[category]_val.txt.
            category = path.basename(annotation_path)[:-len("_val.txt")]
        elif annotation_path.endswith('.txt'):
            # annotation_path is of the form /path/to/[category].txt
            category = path.basename(annotation_path)[:-len(".txt")]
        else:
            assert False, ("Unrecognized form for annotation path %s",
                           annotation_path)
        annotation_details = parse_annotation_file(annotation_path, video_fps)
        for annotation in annotation_details:
            annotation['category'] = category
        annotations.extend(annotation_details)
    with open(args.output_annotation_json, 'wb') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_annotation_dir')
    parser.add_argument(
        'video_frames_info',
        help='CSV of format <video_name>,<fps>[,<num_frames_in_video>]?')
    parser.add_argument('output_annotation_json')

    args = parser.parse_args()

    main()
