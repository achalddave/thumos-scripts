"""Create an LMDB containing LabeledVideoFrames as values.

Takes as input a root directory that contains a subdirectory for each video,
which in turn contain frames for the video. For example:

    <dataset>/
        <video_name>/
            frame1.png
            frame2.png
            ...

The only assumption is that frames are named of the form "frame[0-9]+.png".
The second input is a JSON file containing THUMOS annotations, as output by
parse_temporal_annotations.py.

The output LMDB contains keys "<video_name>-<frame-number>" and corresponding
LabeledVideoFrame as values. For example, video1/frame2.png is stored as the
key "video1-2".
"""

import argparse
import glob
import multiprocessing as mp

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

from util.annotation import load_annotations_json
from video_util import frames_to_lmdb
from video_util import video_frames_pb2


def collect_frame_labels(file_annotations, frame_index, frames_per_second):
    query_second = frame_index * frames_per_second
    return sorted(list(set(annotation.category
                           for annotation in file_annotations
                           if annotation.start_seconds < query_second <
                           annotation.end_seconds)))


def create_labeled_frame(video_name, frame_index, image_proto, labels):
    video_frame = video_frames_pb2.LabeledVideoFrame()
    video_frame.frame.image.CopyFrom(image_proto)
    video_frame.frame.video_name = video_name
    video_frame.frame.frame_index = frame_index
    for label in labels:
        video_frame.label.append(label)
    return video_frame


def load_image(image_path, resize_height=None, resize_width=None):
    """Load an image in video_frames.Image format.

    Args:
        image_path (str): Path to an image.
        resize_height (int): Height to resize an image to. If 0 or None, the
            image is not resized.
        resize_width (int): Width to resize an image to. If 0 or None, the
            image is not resized.

    Returns:
        image_datum (caffe Datum): Contains the image in BGR order after
            resizing.
    """
    image = Image.open(image_path)
    if resize_height and resize_width:
        image = image.resize((resize_width, resize_height))
    # Image has shape (height, width, num_channels), where the
    # channels are in RGB order.
    image = np.array(image)
    # Convert image from RGB to BGR.
    image = image[:, :, ::-1]
    # Convert image to (num_channels, height, width) shape.
    image = image.transpose((2, 0, 1))
    return image


def load_image_helper(args):
    return load_image(*args)


def load_image_batch(pool, frame_paths, resize_height, resize_width):
    """Loads a batch of images by calling load_image_datum in parallel."""
    job_arguments = [(frame_path, resize_height, resize_width)
                     for frame_path in frame_paths]
    return pool.map(load_image_helper, job_arguments)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('frames_root')
    parser.add_argument('annotations_json')
    parser.add_argument('output_lmdb')
    parser.add_argument('--resize_width', default=None, nargs='?', type=int)
    parser.add_argument('--resize_height', default=None, nargs='?', type=int)
    parser.add_argument('--frames_per_second',
                        default=10,
                        help='FPS that frames were extracted at.')

    args = parser.parse_args()

    # TODO(achald): Allow specifying either one, and resize the other based on
    # aspect ratio.
    if (args.resize_width is None) != (args.resize_height is None):
        raise ValueError('Both resize_width and resize_height must be '
                         'specified if either is specified.')
    map_size = int(500e9)

    batch_size = 10000

    # Load pairs of the form (frame path, (video name, frame index)), and
    # create batches..
    frame_path_info_pairs = [
        (frame_path, frames_to_lmdb.parse_frame_path(frame_path))
        for frame_path in glob.iglob('{}/*/*.png'.format(args.frames_root))
    ]

    frame_path_info_pairs_batched = (
        frame_path_info_pairs[i:i + batch_size]
        for i in range(0, len(frame_path_info_pairs), batch_size))
    print 'Loaded frame paths.'

    annotations = load_annotations_json(args.annotations_json)

    progress = tqdm(total=len(frame_path_info_pairs))
    pool = mp.Pool(8)
    for frame_path_info_pairs_batch in frame_path_info_pairs_batched:
        batch_images = load_image_batch(
            pool, [x[0] for x in frame_path_info_pairs_batch],
            args.resize_height, args.resize_height)

        # Convert image arrays to image protocol buffers.
        # We can't return protos from multiprocessing due to pickling issues.
        # https://groups.google.com/forum/#!topic/protobuf/VqWJ3BmQXVg
        for i in range(len(batch_images)):
            image_array = batch_images[i]
            image = video_frames_pb2.Image()
            image.channels, image.height, image.width = image_array.shape
            image.data = image_array.tostring()
            batch_images[i] = image

        video_frames_batch = []
        with lmdb.open(args.output_lmdb, map_size=map_size).begin(
                write=True) as lmdb_transaction:
            for i in range(len(frame_path_info_pairs_batch)):
                video_name, frame_index = frame_path_info_pairs_batch[i][1]
                labels = collect_frame_labels(annotations[video_name],
                                              frame_index - 1,
                                              args.frames_per_second)
                video_frame_proto = create_labeled_frame(
                    video_name, frame_index, batch_images[i], labels)
                frame_key = '{}-{}'.format(video_name, frame_index)
                lmdb_transaction.put(frame_key,
                                     video_frame_proto.SerializeToString())
                progress.update(1)
        del batch_images


if __name__ == "__main__":
    main()
