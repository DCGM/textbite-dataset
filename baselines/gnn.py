# ruff: noqa: E402

from safe_gpu import safe_gpu
safe_gpu.claim_gpus()

import argparse
import os
import logging
import json
import pickle

from pero_ocr.core.layout import PageLayout
from ultralytics import YOLO
import torch
from transformers import BertTokenizerFast, BertModel

from textbite.models.yolo.infer import YoloBiter
from textbite.models.utils import GraphNormalizer
from textbite.models.joiner.graph import JoinerGraphProvider
from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.joiner.infer import join_bites


CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])

    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--img", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--output", required=True, type=str, help="Folder where to put output files.")

    parser.add_argument("--yolo", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--gnn", required=True, type=str, help="Path to the .pt file with weights of Joiner model.")
    parser.add_argument("--normalizer", required=True, type=str, help="Path to node normalizer.")
    parser.add_argument("--czert", default=CZERT_PATH, type=str, help="Path to CZERT.")

    parser.add_argument("--threshold", type=float, required=True, help="Threshold for classification.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YoloBiter(YOLO(args.yolo))

    tokenizer = BertTokenizerFast.from_pretrained(args.czert)
    czert = BertModel.from_pretrained(args.czert)
    czert = czert.to(device)

    graph_provider = JoinerGraphProvider(tokenizer, czert, device)

    gnn_checkpoint = torch.load(args.gnn)
    gnn = JoinerGraphModel.from_pretrained(gnn_checkpoint, device)
    gnn.eval()
    gnn = gnn.to(device)

    with open(args.normalizer, "rb") as f:
        normalizer: GraphNormalizer = pickle.load(f)

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
        except OSError:
            logging.warning(f"XML file {xml_path} not found. SKIPPING")
            continue

        pagexml_line_dict = {line.id: line.polygon for line in pagexml.lines_iterator()}

        logging.info(f"({i+1}/{len(img_filenames)}) | Processing: {xml_path}")

        yolo_bites = yolo.produce_bites(img_path, pagexml)

        try:
            bites = join_bites(
                yolo_bites,
                gnn,
                graph_provider,
                normalizer,
                base_filename,
                pagexml,
                device,
                threshold=args.threshold,
            )
        except RuntimeError:
            logging.info(f"Single region detected on {xml_path}, saving as is.")
            bites = yolo_bites

        result = []

        for bite in bites:
            bite_polygons = []

            for line_id in bite.lines:
                line_polygon = pagexml_line_dict[line_id].tolist()
                bite_polygons.append(line_polygon)

            result.append(bite_polygons)

        with open(json_save_path, "w") as f:
            json.dump(result, f, indent=2)
    

if __name__ == "__main__":
    main()
