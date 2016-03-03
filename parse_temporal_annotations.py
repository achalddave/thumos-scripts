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
import json
import os
from os import path

from util.parsing import parse_video_fps_file, parse_annotation_file


def main():
    annotation_paths = ["%s/%s" % (args.input_annotation_dir, x)
                        for x in os.listdir(args.input_annotation_dir)]
    # Maps video name to frames per second
    video_fps = parse_video_fps_file(args.video_frames_info)
    annotations = []
    for annotation_path in annotation_paths:
        category = path.splitext(path.basename(annotation_path))[0]
        if category.endswith('_val'):
            category = category[:-len('_val')]
        elif category.endswith('_test'):
            category = category[:-len('_test')]
        annotation_details = parse_annotation_file(annotation_path, video_fps,
                                                   category)
        annotations.extend([annotation._asdict()
                            for annotation in annotation_details])
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
