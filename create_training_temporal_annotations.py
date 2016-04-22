"""Create temporal annotations for (trimmed) training videos.

Output format (same as parse_temporal_annotations.py):
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

The training videos don't come with a separate annotations file, but they are
trimmed videos that belong to one action category, and the action category is
contained within the video name. We use this to output the training labels.
"""


import argparse
import json
from os import path

from moviepy.editor import VideoFileClip
from tqdm import tqdm

from util.parsing import load_class_mapping


def video_info(video_path):
    """
    Returns:
        duration_seconds (float): Duration in seconds.
        num_frames (int): Number of frames in the video.
        fps (num): Number of frames in a second. (This is the frame rate of the
            video, and is not necessarily equal to duration_seconds /
            num_frames, although it's not clear when the two would be
            different.)
    """
    clip = VideoFileClip(video_path)
    return (clip.duration, clip.reader.nframes, clip.fps)


def extract_label(video_name):
    """Extract label from video file name.

    >>> extract_label('v_YoYo_g25_c05.avi')
    'YoYo'
    >>> extract_label('v_Knitting_g16_c02.avi')
    'Knitting'
    """
    # Videos are of the form 'v_<ClassName>_<id>.<extension>'
    try:
        return video_name.split('_')[1]
    except:
        raise ValueError('Not a valid video name.')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'training_videos_list',
        help='New-line delimited file containing path to training videos.')
    parser.add_argument(
        'class_mapping',
        help=('File containing lines of the form '
              '"<class_index> <class_name>". The order of lines in this file '
              'will correspond to the order of the labels in the output label '
              'matrix. If a video contains a class not in this file, it will '
              'be ignored.'))
    parser.add_argument('output_annotations_json')

    args = parser.parse_args()
    with open(args.training_videos_list) as f:
        video_paths = [line.strip() for line in f]

    valid_labels = set(load_class_mapping(args.class_mapping).values())
    annotations = []
    for video_path in tqdm(video_paths):
        if video_path[-1] == '/': video_path = video_path[:-1]
        video_name = path.splitext(path.basename(video_path))[0]

        try:
            label = extract_label(video_name)
            if label not in valid_labels:
                continue
        except ValueError:
            continue

        duration, num_frames, fps = video_info(video_path)
        annotations.append({
            'filename': video_name,
            'start_frame': 0,
            'end_frame': num_frames,
            'start_seconds': 0,
            'end_seconds': duration,
            'frames_per_second': fps,
            'category': label
        })

    with open(args.output_annotations_json, 'wb') as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    main()
