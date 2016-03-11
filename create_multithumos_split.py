"""Split MultiTHUMOS annotations into val and test for evaluation."""

import argparse
import os
from os import path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('multithumos_annotations_dir')
    parser.add_argument('output_dir')

    args = parser.parse_args()
    root = args.multithumos_annotations_dir
    annotation_files = [path.join(root, x)
                        for x in os.listdir(root) if x.endswith('.txt')]

    val_dir = path.join(args.output_dir, 'val')
    test_dir = path.join(args.output_dir, 'test')
    if not path.isdir(args.output_dir): os.mkdir(args.output_dir)
    if not path.isdir(val_dir): os.mkdir(val_dir)
    if not path.isdir(test_dir): os.mkdir(test_dir)

    for annotation_file in annotation_files:
        filename = path.splitext(path.basename(annotation_file))[0]
        validation_lines = []
        test_lines = []
        with open(annotation_file) as f:
            for line in f:
                video_filename = line.split(' ')[0]
                if video_filename.startswith('video_validation_'):
                    validation_lines.append(line)
                elif video_filename.startswith('video_test_'):
                    test_lines.append(line)
                else:
                    raise ValueError('Unknown file split %s' % video_filename)
        validation_output_file = path.join(val_dir, filename + '_val.txt')
        test_output_file = path.join(test_dir, filename + '_test.txt')
        with open(validation_output_file, 'wb') as f:
            f.writelines(validation_lines)
        with open(test_output_file, 'wb') as f:
            f.writelines(test_lines)
    # Create empty ambigious files which the THUMOS eval script looks for.
    open(path.join(test_dir, 'Ambiguous_test.txt'), 'w').close()
    open(path.join(val_dir, 'Ambiguous_val.txt'), 'w').close()
