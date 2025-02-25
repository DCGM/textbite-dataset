#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import os
import json
from tqdm import tqdm

import sklearn.metrics

from common import get_hypothesis_segmentation


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hyp-dir", type=str, required=True, help="Directory with prediction JSON files")
    parser.add_argument("--ref-dir", type=str, required=True, help="Directory with reference numpy files")

    return parser.parse_args()


def score_bites_segmentation(
    reference,
    hypothesis,
) -> float:
    assert reference.shape == hypothesis.shape
    reference_mask = reference != 0

    hypothesis_relevant_pixels = hypothesis[reference_mask]
    reference_relevant_pixels = reference[reference_mask]

    return sklearn.metrics.rand_score(reference_relevant_pixels, hypothesis_relevant_pixels)


def main():
    args = parse_arguments()

    ref_filenames = [filename for filename in os.listdir(args.ref_dir) if filename.endswith(".npy")]
    nb_paired = 0
    total_rand = 0.0

    for filename in tqdm(ref_filenames):
        with open(os.path.join(args.ref_dir, filename), "rb") as f:
            reference = np.load(f)

        try:
            hyp_fn = filename[:-len(".npy")] + ".json"
            with open(os.path.join(args.hyp_dir, hyp_fn), 'r') as f:
                hypothesis = json.load(f)
        except FileNotFoundError:
            logging.warning(f'No hypothesis found for {filename}')
            continue

        nb_paired += 1
        hypothesis_segmentation = get_hypothesis_segmentation(reference.shape, hypothesis)
        total_rand += score_bites_segmentation(reference, hypothesis_segmentation)

    print(f'Average Rand score: {total_rand/nb_paired}, computed from {nb_paired} pages')
    if nb_paired < len(ref_filenames):
        logging.warning(f'Only {nb_paired} out of {len(ref_filenames)} pairs were compared ({nb_paired/len(ref_filenames)*100.0:.2f}) %)')


if __name__ == '__main__':
    main()
