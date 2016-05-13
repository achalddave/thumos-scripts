"""Split THUMOS' validation set into 'trainval' and 'valval' for MultiTHUMOS.

MultiTHUMOS annotations are not available on the training videos of THUMOS.
Instead, we split up the validation subset further into a 'train' and 'val'
subset ('trainval' and 'valval').

For each category, we pick X% (rounded up) of the videos and place them in the
'valval' set.
"""

import argparse
import logging
import random

from util.annotation import (filter_annotations_by_category,
                             load_annotations_json)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                    datefmt='%H:%M:%S')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('val_annotations_json')
    parser.add_argument('output_trainval_names',
                        help='File to output trainval video names to.')
    parser.add_argument('output_valval_names',
                        help='File to output valval video names to.')
    parser.add_argument(
        '--val_portion',
        default=0.2,
        type=float,
        help='Minimum proportion of videos for each category in valval set.')
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    random.seed(args.seed)

    val_annotations = load_annotations_json(args.val_annotations_json)
    categories = set()
    for file_annotations in val_annotations.values():
        categories.update(x.category for x in file_annotations)

    valval_videos = set()
    # Sort to ensure deterministic order conditioned on seed.
    for category in sorted(list(categories)):
        category_annotations = filter_annotations_by_category(val_annotations,
                                                              category)
        video_names = category_annotations.keys()
        in_valval = set(video_names).intersection(valval_videos)
        num_valval = max(int(round(args.val_portion * len(video_names))), 1)
        if len(in_valval) >= num_valval:
            continue

        # Sort to ensure deterministic order conditioned on seed.
        not_in_valval = sorted(list(set(video_names) - valval_videos))
        random.shuffle(not_in_valval)
        valval_videos.update(not_in_valval[:num_valval - len(in_valval)])

    valtrain_videos = set(val_annotations.keys()) - valval_videos
    logging.info('# train videos: %s', len(valtrain_videos))
    logging.info('# val videos: %s', len(valval_videos))
    with open(args.output_trainval_names, 'wb') as train_f, \
            open(args.output_valval_names, 'wb') as val_f:
        train_f.writelines([x + '\n' for x in valtrain_videos])
        val_f.writelines([x + '\n' for x in valval_videos])

if __name__ == "__main__":
    main()
