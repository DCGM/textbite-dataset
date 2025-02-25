import numpy as np
import skimage.draw


def get_bite_mask(
    img_shape: tuple[int, int],
    list_of_polygons: list[list[list[float | int]]],
) -> np.ndarray:
    bite_mask = np.zeros(img_shape, dtype=np.bool)
    for p in list_of_polygons:
        bite_mask += skimage.draw.polygon2mask(bite_mask.shape[::-1], p).T

    return bite_mask


def get_hypothesis_segmentation(
    img_shape: tuple[int, int],
    hypothesis: list[list[list[list[float | int]]]],
) -> np.ndarray:
    assert len(hypothesis) < 256

    segmentation = np.zeros(img_shape, dtype=np.uint8)
    for i, bite in enumerate(hypothesis, start=1):
        segmentation += (i * get_bite_mask(img_shape, bite)).astype(np.uint8)

    return segmentation
