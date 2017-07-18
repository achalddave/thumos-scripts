"""Parse temporal annotations for THUMOS '14 validation data, output JSON.

Output format:
    [
        {
            filename: str,
            start_seconds: float,
            end_seconds: float,
            start_frame: int,
            end_frame: int,
            frames_per_second: float,
            category: str
        },
        ...
    ]
"""

import argparse
import json

from util.parsing import load_thumos_annotations


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_annotation_dir')
    parser.add_argument(
        'video_frames_info',
        help='CSV of format <video_name>,<fps>[,<num_frames_in_video>]?')
    parser.add_argument('output_annotation_json')

    args = parser.parse_args()

    annotations = [
        annotation._asdict()
        for annotation in load_thumos_annotations(args.input_annotation_dir,
                                                  args.video_frames_info)
    ]
    with open(args.output_annotation_json, 'wb') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    main()
