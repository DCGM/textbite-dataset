import argparse
import logging
from tqdm import tqdm

import os
import json
import numpy as np
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


def bootstrap(
    scores,
    n_resamples,
    confidence=0.95,
):
    boot_means = [np.mean(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_resamples)]
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return np.mean(boot_means), lower, upper


def main():
    args = parse_arguments()

    ref_filenames = [filename for filename in os.listdir(args.ref_dir) if filename.endswith(".npy")]
    nb_paired = 0
    total_rand = 0.0
    scores = []

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
        score = score_bites_segmentation(reference, hypothesis_segmentation)
        total_rand += score
        scores.append(score)

    print(f'Average Rand score: {total_rand/nb_paired}, computed from {nb_paired} pages')

    mean, lower_bound, upper_bound = bootstrap(scores, 1000)
    print(f'Bootstrap confidence interval: {mean} [{lower_bound}, {upper_bound}]')
    if nb_paired < len(ref_filenames):
        logging.warning(f'Only {nb_paired} out of {len(ref_filenames)} pairs were compared ({nb_paired/len(ref_filenames)*100.0:.2f}) %)')


if __name__ == '__main__':
    main()
