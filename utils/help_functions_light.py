import numpy as np
import cv2
import os
import math
from shapely import Polygon, unary_union
from utils import ml_config
from utils import config
import datetime

def is_im_path(im_path, suffixes=['jpg', 'tiff', 'png', 'jpeg', 'tif']):
    for s in suffixes:
        if im_path.endswith(s):
            return True
    return False


def calc_width_parts(img_width, frag_size):
    if frag_size / 2 > img_width:
        return [[0, img_width]]
    crop_start_end_coords = []
    tek_pos = 0
    while tek_pos <= img_width:
        if tek_pos == 0:
            if img_width > frag_size:
                crop_start_end_coords.append([tek_pos, frag_size])
            else:
                crop_start_end_coords.append([tek_pos, img_width])
                break

        elif tek_pos + frag_size >= img_width:
            crop_start_end_coords.append([tek_pos, img_width])
            break

        else:
            crop_start_end_coords.append([tek_pos, tek_pos + frag_size])
        tek_pos += int(frag_size / 2)

    return crop_start_end_coords


def density_slider_to_value(value, min_value=config.MIN_DENSITY_VALUE, max_value=config.MAX_DENSITY_VALUE):
    b = 0.01 * math.log(max_value / min_value)
    return min_value * math.exp(b * value)


def calc_parts(img_width, img_height, frag_size):
    crop_x_y_sizes = []
    crop_x_sizes = calc_width_parts(img_width, int(frag_size))
    crop_y_sizes = calc_width_parts(img_height, int(frag_size))
    for y in crop_y_sizes:
        for x in crop_x_sizes:
            crop_x_y_sizes.append([x, y])
    return crop_x_y_sizes, len(crop_x_sizes), len(crop_y_sizes)


def split_into_fragments(img, frag_size):
    fragments = []

    shape = img.shape

    img_width = shape[1]
    img_height = shape[0]

    crop_x_y_sizes, x_parts_num, y_parts_num = calc_parts(img_width, img_height, frag_size)

    for x_y_crops in crop_x_y_sizes:
        x_min, x_max = x_y_crops[0]
        y_min, y_max = x_y_crops[1]
        fragments.append(img[int(y_min):int(y_max), int(x_min):int(x_max), :])

    return fragments


def convert_image_name_to_png_name(image_name):
    splitted_name = image_name.split('.')
    txt_name = ""
    for i in range(len(splitted_name) - 1):
        txt_name += splitted_name[i]

    return txt_name + ".png"


def convert_item_polygon_to_point_mass(pol):
    points = []
    for p in pol:
        points.append([p.x(), p.y()])
    return points


def calc_label_pos(point_mass):
    p = Polygon(point_mass)
    c = p.boundary.coords[1]
    return [int(c[0]), int(c[1])]


def calc_rows_cols(size):
    rows = min(1, math.ceil(math.sqrt(size)))
    cols = int(size / rows)
    assert rows * cols == size

    return rows, cols


def save_mask_as_image(mask, save_name):
    height, width = mask.shape
    im = np.zeros((width, height))
    im[mask] = 255
    cv2.imwrite(save_name, im)


def handle_temp_folder(cwd):
    if not cwd:
        cwd = os.getcwd()

    temp_folder = os.path.join(cwd, 'temp')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    return temp_folder


def create_unique_image_name(image_name):
    splitted_name = image_name.split('.')
    new_name = ""
    for i in range(len(splitted_name) - 1):
        new_name += splitted_name[i]

    return f'{new_name} {datetime.datetime.now().microsecond}.{splitted_name[-1]}'
