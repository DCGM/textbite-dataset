import argparse
import json
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.draw

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

    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].imshow(reference)
    ax[1].set_title("Ground truth")
    ax[1].axis("off")

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

    ax[2].imshow(img_hypothesis)
    ax[2].set_title("Hypothesis")
    ax[2].axis("off")

    hyp_segmentation = get_hypothesis_segmentation(reference.shape, hypothesis)# & reference
    ax[3].imshow(hyp_segmentation)
    ax[3].set_title("Hypothesis segmentation")
    ax[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close()


def main():
    args = parse_arguments()

    hyp_filenames = [f for f in os.listdir(args.hyp) if f.endswith(".json")]
    gt_filenames = [f for f in os.listdir(args.ref) if f.endswith(".npy")]

    hyp_filenames = list(set([f[:-5] for f in hyp_filenames]) & set([f[:-4] for f in gt_filenames]))
    hyp_filenames = [f + ".json" for f in hyp_filenames]
    os.makedirs(args.output, exist_ok=True)

    for hyp_fn in hyp_filenames:
        hyp_path = os.path.join(args.hyp, hyp_fn)

        ref_fn = hyp_fn.replace(".json", ".npy")
        ref_path = os.path.join(args.ref, ref_fn)

        img_fn = hyp_fn.replace(".json", ".jpg")
        img_path = os.path.join(args.img, img_fn)

        output_fn = hyp_fn.replace(".json", ".pdf")
        output_path = os.path.join(args.output, output_fn)

        process_file(hyp_path, ref_path, img_path, output_path)


if __name__ == "__main__":
    main()
