# ruff: noqa: E402

from safe_gpu import safe_gpu
safe_gpu.claim_gpus()

import argparse
import os
import logging
import json
from enum import Enum

import torch
from ultralytics import YOLO
from pero_ocr.core.layout import PageLayout
import cv2
import numpy as np

from textbite.models.yolo.infer import YoloBiter
from textbite.bite import Bite

from organizer.utils import load_image
from organizer.model import MultimodalLayoutTransformerConfig, MultimodalLayoutTransformer


class QueryType(Enum):
    PADDING = 0
    BBOX = 1
    CLASS_QUERY = 2


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])

    parser.add_argument("--yolo", type=str, required=True, default="Path to the .pt file with weights of YOLO model")
    parser.add_argument("--transformer", type=str, required=True, default="Path to a pth file with a trained model")
    parser.add_argument("--config", type=str, required=True, default="Path to a json file with a model configuration")

    parser.add_argument("--img", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--output", type=str, required=True, default="Path to a folder where to save the output")

    return parser.parse_args()


def join_bites(
    bites: list[Bite],
    transformer: MultimodalLayoutTransformer,
    image_path: str,
) -> list[Bite]:
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    bboxes = np.array([[bite.bbox.xmin / image_width, bite.bbox.ymin / image_height, bite.bbox.xmax / image_width, bite.bbox.ymax / image_height] for bite in bites], dtype=np.float32)
    bboxes = np.array(bboxes, dtype=np.float32)
    if bboxes.shape[0] > transformer.config.max_bbox_count:
        bboxes = bboxes[:transformer.config.max_bbox_count]
    bboxes = np.clip(bboxes, a_min=0.0, a_max=1.0)
    query_types = np.array([QueryType.BBOX.value] * len(bboxes) + [QueryType.PADDING.value] * (transformer.config.max_bbox_count - len(bboxes)))
    bboxes = np.pad(
        bboxes,
        ((0, transformer.config.max_bbox_count - bboxes.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    bboxes = torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0).to(transformer.device)
    query_types = torch.tensor(query_types, dtype=torch.long).unsqueeze(0).to(transformer.device)

    image_ = load_image(image_path)
    image_ = cv2.resize(image_, (transformer.config.image_height, transformer.config.image_width))
    image_ = image_.transpose(2, 0, 1)
    image_ = torch.tensor(image_, dtype=torch.uint8).unsqueeze(0).to(transformer.device)

    with torch.no_grad():
        predictions = transformer(
            x=bboxes,
            query_types=query_types,
            images=image_,
            input_ids=None,
            attention_mask=None,
        ).bbox_similarity_matrix

    predictions = predictions[:, :len(bites), :len(bites)]
    successors = predictions[0].argmax(dim=1).cpu().numpy().tolist()

    for _ in range(len(successors) + 1):
        for from_idx, to_idx in enumerate(successors):
            if from_idx == to_idx:
                continue

            bites[to_idx].lines.extend(bites[from_idx].lines)
            bites[from_idx].lines = []

    bites = [bite for bite in bites if bite.lines]

    return bites


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YoloBiter(YOLO(args.yolo))

    model_config = MultimodalLayoutTransformerConfig.from_file(args.config)
    model_checkpoint = torch.load(args.transformer, weights_only=True)
    model = MultimodalLayoutTransformer(
        config=model_config,
        device=device,
    )
    model.load_state_dict(model_checkpoint)
    model = model.to(device)
    model.eval()

    print(model)

    os.makedirs(args.output, exist_ok=True)

    img_filenames = [img_filename for img_filename in os.listdir(args.img) if img_filename.endswith(".jpg")]
    for i, img_filename in enumerate(img_filenames):
        base_filename = img_filename[:-len(".jpg")]
        xml_filename = base_filename + ".xml"
        json_filename = base_filename + ".json"

        img_path = os.path.join(args.img, img_filename)
        xml_path = os.path.join(args.xml, xml_filename)
        json_save_path = os.path.join(args.output, json_filename)

        try:
            pagexml = PageLayout(file=xml_path)
            pagexml_line_dict = {line.id: line.polygon for line in pagexml.lines_iterator()}
        except OSError:
            logging.warning(f"XML file {xml_path} not found. SKIPPING")
            continue

        bites = yolo.produce_bites(img_path, pagexml)
        yolo_bites_n = sum([len(bite.lines) for bite in bites])

        if len(bites) > 1:
            bites = join_bites(
                bites=bites,
                transformer=model,
                image_path=img_path,
            )

        joined_bites_n = sum([len(bite.lines) for bite in bites])
        assert yolo_bites_n == joined_bites_n
        
        result = []
        for bite in bites:
            bite_polygons = []

            for line_id in bite.lines:
                line_polygon = pagexml_line_dict[line_id].tolist()
                bite_polygons.append(line_polygon)

            result.append(bite_polygons)

        with open(json_save_path, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
