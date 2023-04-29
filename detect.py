import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm
import numpy as np

def get_size(image):
    window_height = image.shape[0]
    window_width = image.shape[1]
    return window_height * window_width * 0.0001

def get_color_amount(minimum_shape_size, mask):
    counter = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        size = cv2.contourArea(contour)
        if size > minimum_shape_size:
            counter += 1
    return counter

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    counter = [0, 0, 0, 0]
    minimum_shape_size = get_size(image)
    lower_hsv_green = np.array([36, 100, 100])
    upper_hsv_green = np.array([86, 255, 255])
    lower_hsv_red = np.array([166, 100, 100])
    upper_hsv_red = np.array([190, 255, 255])
    lower_hsv_yellow = np.array([24, 100, 100])
    upper_hsv_yellow = np.array([30, 255, 255])
    lower_hsv_purple = np.array([147, 50, 50])
    upper_hsv_purple = np.array([169, 255, 255])

    green_mask = cv2.inRange(hsv_image, lower_hsv_green, upper_hsv_green)
    counter[2] = get_color_amount(minimum_shape_size, green_mask)
    red_mask = cv2.inRange(hsv_image, lower_hsv_red, upper_hsv_red)
    counter[0] = get_color_amount(minimum_shape_size, red_mask)
    yellow_mask = cv2.inRange(hsv_image, lower_hsv_yellow, upper_hsv_yellow)
    counter[1] = get_color_amount(minimum_shape_size, yellow_mask)
    purple_mask = cv2.inRange(hsv_image, lower_hsv_purple, upper_hsv_purple)
    counter[3] = get_color_amount(minimum_shape_size, purple_mask)

    return {'red': counter[0], 'yellow': counter[1], 'green': counter[2], 'purple': counter[3]}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)


def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
