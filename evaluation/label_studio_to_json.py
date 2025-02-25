#!/usr/bin/env python3

import argparse
import json
import logging
import os

import detector_wrapper.parsers.detector_parser
from produce_labeling import organize_bboxes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_studio_export')
    parser.add_argument('output_folder')

    return parser.parse_args()


def bbox_to_polygon(bbox):
    x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
    return [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ]


def group_into_list(bboxes):
    return [bbox_to_polygon(bbox) for bbox in bboxes]


def main():
    args = get_args()
    os.makedirs(args.output_folder, exist_ok=True)

    parser = detector_wrapper.parsers.detector_parser.DetectorParser()
    parser.parse_label_studio(args.label_studio_export, run_checks=True)

    nb_failed_pages = 0
    for page in parser.annotated_pages:
        try:
            grouped_bboxes = organize_bboxes(page)
            bboxes_dict = {bbox.id: bbox for bbox in page.bounding_boxes}

            list_of_bites = [group_into_list([bboxes_dict[i] for i in group]) for group in grouped_bboxes]

            out_name = page.image_filename.split('/')[-1][:-4] + '.json'
            with open(os.path.join(args.output_folder, out_name), 'w') as f:
                json.dump(list_of_bites, f, indent=4, ensure_ascii=False)

        except Exception:
            logging.error(f'Failed to parse annotation for page {page.image_filename}')
            nb_failed_pages += 1

    if nb_failed_pages > 0:
        logging.warning(f'There were {nb_failed_pages} pages that failed to parse ({nb_failed_pages/len(parser.annotated_pages)*100.0:.2f} %)')


if __name__ == '__main__':
    main()
