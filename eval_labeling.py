#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import os
import json

import skimage.draw
import sklearn.metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp_dir')
    parser.add_argument('ref_dir')

    return parser.parse_args()


def get_bite_mask(img_shape, list_of_polygons):
    bite_mask = np.zeros(img_shape, dtype=np.bool)
    for p in list_of_polygons:
        bite_mask += skimage.draw.polygon2mask(bite_mask.shape[::-1], p).T

    return bite_mask


def get_hypothesis_segmentation(img_shape, hypothesis):
    assert len(hypothesis) < 256
    segmentation = np.zeros(img_shape, dtype=np.uint8)
    for i, bite in enumerate(hypothesis, start=1):
        segmentation += (i * get_bite_mask(img_shape, bite)).astype(np.uint8)

    return segmentation


def score_bites_segmentation(reference, hypothesis):
    assert reference.shape == hypothesis.shape
    reference_mask = reference != 0

    hypothesis_relevant_pixels = hypothesis[reference_mask]
    reference_relevant_pixels = reference[reference_mask]

    return sklearn.metrics.rand_score(reference_relevant_pixels, hypothesis_relevant_pixels)


def main():
    args = get_args()

    ref_fns = [fn for fn in os.listdir(args.ref_dir) if fn.endswith('.npy')]
    nb_paired = 0
    total_rand = 0.0
    for fn in ref_fns:
        with open(os.path.join(args.ref_dir, fn), 'rb') as f:
            reference = np.load(f)

        try:
            hyp_fn = fn[:-len('.npy')] + '.json'
            with open(os.path.join(args.hyp_dir, hyp_fn), 'r') as f:
                hypothesis = json.load(f)
        except FileNotFoundError:
            logging.warning(f'No hypothesis found for {fn}')
            continue

        nb_paired += 1
        hypothesis_segmentation = get_hypothesis_segmentation(reference.shape, hypothesis)
        total_rand += score_bites_segmentation(reference, hypothesis_segmentation)

    print(f'Average Rand score: {total_rand/nb_paired}, computed from {nb_paired} pages')
    if nb_paired < len(ref_fns):
        logging.warning(f'Only {nb_paired} out of {len(ref_fns)} pairs were compared ({nb_paired/len(ref_fns)*100.0:.2f}) %)')


if __name__ == '__main__':
    main()
