import argparse
import json
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import get_hypothesis_segmentation


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hyp-dir", type=str, required=True, help="Path to the prediction folder.")
    parser.add_argument("--ref-dir", type=str, required=True, help="Path to the reference folder.")
    parser.add_argument("--img-dir", type=str, required=True, help="Path to the image folder.")
    parser.add_argument("--out-dir", type=str, required=True, help="Path to the output folder.")

    return parser.parse_args()


def process_file(
    hyp_path: str,
    ref_path: str,
    img_path: str,
    output_path: str,
) -> None:
    with open(hyp_path, "r") as f:
        hypothesis = json.load(f)

    reference = np.load(ref_path)
    image = cv2.imread(img_path)

    fig, ax = plt.subplots(1, 4)

    # Calculate hypothesis segmentation
    img_hypothesis = image.copy()
    img_overlay = image.copy()
    alpha = 0.35
    for bite_idx, bite in enumerate(hypothesis):
        color = (255, 0, 0)
        diff = 255 // len(hypothesis)
        color = (color[0] - diff * bite_idx, color[1] + diff * bite_idx, color[2] + diff * bite_idx)

        for polygon in bite:
            polygon = np.array(polygon)
            polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(img_overlay, [polygon], color)
    img_hypothesis = cv2.addWeighted(img_overlay, alpha, image, 1 - alpha, 0, image)

    # Plot original image
    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[0].axis("off")

    max_bites = max((int(reference.max()) + 1, int(img_overlay.max()) + 1))
    cmap = plt.get_cmap('viridis', max_bites)

    # Plot ground truth
    ax[1].imshow(reference, cmap=cmap)
    ax[1].set_title("Ground truth")
    ax[1].axis("off")

    # Plot predictions
    ax[2].imshow(img_hypothesis, cmap=cmap)
    ax[2].set_title("Hypothesis")
    ax[2].axis("off")

    # Plot prediction segmentation
    hyp_segmentation = get_hypothesis_segmentation(reference.shape, hypothesis)
    ax[3].imshow(hyp_segmentation, cmap=cmap)
    ax[3].set_title("Hypothesis segmentation")
    ax[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close()


def main():
    args = parse_arguments()

    hyp_filenames = [f for f in os.listdir(args.hyp_dir) if f.endswith(".json")]
    gt_filenames = [f for f in os.listdir(args.ref_dir) if f.endswith(".npy")]

    hyp_filenames = list(set([f[:-5] for f in hyp_filenames]) & set([f[:-4] for f in gt_filenames]))
    hyp_filenames = [f + ".json" for f in hyp_filenames]
    os.makedirs(args.out_dir, exist_ok=True)

    for hyp_fn in hyp_filenames:
        hyp_path = os.path.join(args.hyp_dir, hyp_fn)

        ref_fn = hyp_fn.replace(".json", ".npy")
        ref_path = os.path.join(args.ref_dir, ref_fn)

        img_fn = hyp_fn.replace(".json", ".jpg")
        img_path = os.path.join(args.img_dir, img_fn)

        output_fn = hyp_fn.replace(".json", ".jpg")
        output_path = os.path.join(args.out_dir, output_fn)

        process_file(hyp_path, ref_path, img_path, output_path)


if __name__ == "__main__":
    main()
