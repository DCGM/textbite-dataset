import os
import json
import argparse
from tqdm import tqdm

from ultralytics import YOLO
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--images", type=str, required=True, help="Path to the folder with images")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output folder")

    return parser.parse_args()


def bbox_to_polygon(bbox):
    x1, y1, x2, y2 = bbox

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]


def main():
    args = parse_arguments()

    os.makedirs(args.output, exist_ok=True)
    
    model = YOLO(args.model)
    image_filenames = [f for f in os.listdir(args.images) if f.endswith(".jpg")]
    
    for img_name in tqdm(image_filenames):
        img_path = os.path.join(args.images, img_name)
        json_path = os.path.join(args.output, f"{os.path.splitext(img_name)[0]}.json")
        
        results = model(img_path)
        
        all_bites = []
        detections = results[0].boxes.xyxy.cpu().numpy()

        for detected_bbox in detections:
            bite = [] # list of polygons
            polygon = bbox_to_polygon(detected_bbox[:4]) # list of points
            bite.append(polygon)
            all_bites.append(bite)
        
        with open(json_path, "w") as f:
            json.dump(all_bites, f, indent=4)


if __name__ == "__main__":
    main()
