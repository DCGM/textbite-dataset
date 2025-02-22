#!/usr/bin/env python3

import argparse
import logging
import os
import numpy as np
import cv2
import skimage.draw

import matplotlib.pyplot as plt
from pero_ocr.core.layout import PageLayout
import detector_wrapper.parsers.detector_parser


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--show-plots', action='store_true', help='Show plots')
    parser.add_argument('--save-plots', help='Where to put plots. If not given, they are not saved.')

    parser.add_argument('export', help='LabelStudio export JSON')
    parser.add_argument('img_dir', help='Folder with images. Driving one.')
    parser.add_argument('xml_dir', help='Folder with PageXMLs describing pages. Which are used is given by images.')
    parser.add_argument('out_dir', help='Where to put the final annotations.')

    return parser.parse_args()


def organize_bboxes(annotated_page):
    bboxes = {bbox.id: bbox for bbox in annotated_page.bounding_boxes}
    bbox_keys_to_org = set(bboxes.keys())

    grouped_bboxes = [set([bbox_id]) for bbox_id in bbox_keys_to_org]

    for relation in annotated_page.relations:
        if relation.from_id == relation.to_id:
            logging.warning(f'Found self-relation for bbox {relation.from_id}')
            continue

        def locate_bbox(bbox_id):
            for i, group in enumerate(grouped_bboxes):
                if bbox_id in group:
                    return i

        from_match = locate_bbox(relation.from_id)
        to_match = locate_bbox(relation.to_id)

        assert from_match is not None
        assert to_match is not None

        grouped_bboxes[from_match].update(grouped_bboxes[to_match])
        del grouped_bboxes[to_match]

    return grouped_bboxes


def get_file_uuid_like_key(path):
    return path.split('/')[-1].split('.')[0]


def get_one_textbite_mask(group_of_bboxes, img_shape):
    label_studio_mask = np.zeros(img_shape, dtype=np.uint8)
    for bbox in group_of_bboxes:
        x_0 = int(bbox.x)
        x_1 = int(bbox.x + bbox.width)
        y_0 = int(bbox.y)
        y_1 = int(bbox.y + bbox.height)
        label_studio_mask[y_0:y_1, x_0:x_1] = 1

    return label_studio_mask


def get_thresholded_mask(img):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img_thresholded_otsu = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thresholded = cv2.adaptiveThreshold(img_grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 301, 2)
    return (1-img_thresholded.astype(bool)).astype(bool)


def get_textline_mask(img_shape, layout):
    textline_mask = np.zeros(img_shape, dtype=np.bool)
    for i, textline in enumerate(layout.lines_iterator()):
        textline_mask += skimage.draw.polygon2mask(textline_mask.shape[::-1], textline.polygon).T

    return textline_mask


def get_label_studio_mask(annotation, groups, img_shape):
    assert len(groups) < 256, 'Too many groups to encode in a single channel'
    bboxes_dict = {bbox.id: bbox for bbox in annotation.bounding_boxes}
    complete_label_studio_mask = np.zeros(img_shape, dtype=np.uint8)
    for i, group in enumerate(groups, start=1):
        group_pixels = get_one_textbite_mask([bboxes_dict[bbox_id] for bbox_id in group], img_shape)
        available_pixels = complete_label_studio_mask == 0
        if group_pixels[complete_label_studio_mask != 0].any():
            logging.warning(f'Overlapping bounding boxes in group {i}')
        group_pixels[complete_label_studio_mask != 0] = 0
        complete_label_studio_mask += i*group_pixels

    return complete_label_studio_mask


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)

    parser = detector_wrapper.parsers.detector_parser.DetectorParser()
    parser.parse_label_studio(args.export, run_checks=True)

    nb_failed_pages = 0
    all_data = {}
    for page in parser.annotated_pages:
        grouped_bboxes = organize_bboxes(page)
        all_data[get_file_uuid_like_key(page.image_filename)] = (page, grouped_bboxes)


    if nb_failed_pages > 0:
        logging.warning(f'There were {nb_failed_pages} pages that failed to parse ({nb_failed_pages/len(parser.annotated_pages)*100.0:.2f} %)')

    nb_img_files_without_annotation = 0
    nb_img_files_without_xml = 0
    img_fns = [fn for fn in os.listdir(args.img_dir) if fn.endswith('.jpg')]
    for fn in img_fns:
        try:
            annotation, groups = all_data[get_file_uuid_like_key(fn)]
        except KeyError:
            logging.error(f'No annotation found for image file {fn}')
            nb_img_files_without_annotation += 1
            continue

        xml_fn = fn[:-len('.jpg')] + '.xml'
        try:
            with open(os.path.join(args.xml_dir, xml_fn)) as f:
                layout = PageLayout(file=f)
        except FileNotFoundError:
            logging.error(f'No PageXML file found for image file {fn}')
            nb_img_files_without_xml += 1
            continue

        img = cv2.imread(os.path.join(args.img_dir, fn))
        img_shape = img.shape[:2]  # drop channel

        complete_label_studio_mask = get_label_studio_mask(annotation, groups, img_shape)
        textline_mask = get_textline_mask(img_shape, layout)
        img_thresholded_otsu = get_thresholded_mask(img)

        complete_annotation = complete_label_studio_mask * textline_mask * img_thresholded_otsu

        with open(os.path.join(args.out_dir, fn[:-len('.jpg')] + '.npy'), 'wb') as f:
            np.save(f, complete_annotation, allow_pickle=False)

        if args.show_plots or args.save_plots:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(img)
            ax[0].set_title('Label studio annotation')
            ax[1].imshow(complete_label_studio_mask)
            ax[1].set_title('Raw label studio annotation')
            ax[2].imshow(complete_annotation)
            ax[2].set_title('Final annotation')

            if args.show_plots:
                fig.canvas.manager.set_window_title(fn)
                plt.show()

            if args.save_plots:
                plt.savefig(os.path.join(args.save_plots, fn[:-len('.jpg')] + '.segmentation.png'), dpi=600)

    if nb_img_files_without_annotation > 0:
        logging.warning(f'There were {nb_img_files_without_annotation} images do not have an annotation ({nb_img_files_without_annotation/len(img_fns)*100.0:.2f} %)')

    if nb_img_files_without_xml > 0:
        logging.warning(f'There were {nb_img_files_without_xml} images do not have an XML file ({nb_img_files_without_xml/len(img_fns)*100.0:.2f} %)')


if __name__ == '__main__':
    main()
