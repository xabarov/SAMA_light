import datetime
import math
import os
import shutil

import cv2
import numpy as np
import screeninfo
import yaml
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtGui import QPolygonF
from shapely import Polygon, unary_union

from utils import ml_config
from utils import config
from utils import coords_calc
from rasterio import features


def save_mask_as_image(mask, save_name):
    height, width = mask.shape
    im = np.zeros((width, height))
    im[mask] = 255
    cv2.imwrite(save_name, im)


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


def calc_rows_cols(size):
    rows = min(1, math.ceil(math.sqrt(size)))
    cols = int(size / rows)
    assert rows * cols == size

    return rows, cols


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




def convert_item_polygon_to_shapely(pol):
    """
    Конвертер полигона QPolygonF to shapely.Polygon
    """
    points = convert_item_polygon_to_point_mass(pol)
    return Polygon(points)


def check_polygon_out_of_screen(pol, width, height):
    for p in pol:
        if p.x() < 0 or p.x() > width:
            return False
        if p.y() < 0 or p.y() > height:
            return False
    return True


def convert_shapely_to_item_polygon(pol):
    coords = pol.exterior.coords
    shapely_pol = QPolygonF()
    for c in coords:
        shapely_pol.append(QtCore.QPointF(c[0], c[1]))
    return shapely_pol


def make_shapely_box(width, height):
    box = []
    box.append([0, 0])
    box.append([width, 0])
    box.append([width, height])
    box.append([0, height])

    box = Polygon(box)

    return box


def merge_polygons(polygons):
    if len(polygons) == 1:
        return polygons

    for i in range(len(polygons)):
        pola = polygons[i]
        for j in range(i + 1, len(polygons)):
            polb = polygons[j]
            if pola.intersects(polb):
                merged_pol = unary_union([pola, polb])
                polygons_new = [polygons[c] for c in range(len(polygons)) if c != i and c != j]
                polygons_new.append(merged_pol)
                return merge_polygons(polygons_new)

    return polygons


def is_polygon_self_intersected(pol):
    pol_shapely = convert_item_polygon_to_shapely(pol)
    if not pol_shapely.is_valid:
        return True
    return False


def try_read_lrm(image_name, from_crs='epsg:3395', to_crs='epsg:4326'):
    ext = coords_calc.get_ext(image_name)
    name_without_ext = image_name[:-len(ext)]
    for map_ext in ['dat', 'map']:
        map_name = name_without_ext + map_ext
        if os.path.exists(map_name):
            coords_net = coords_calc.load_coords(map_name)
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(image_name)

            img_width, img_height = img.size

            return coords_calc.get_lrm(coords_net, img_height)

    if ext == 'tif' or ext == 'tiff':
        return coords_calc.lrm_from_pil_data(image_name, from_crs=from_crs, to_crs=to_crs)


def clear_temp_folder(cwd=None):
    if not cwd:
        cwd = os.getcwd()
    temp_folder = os.path.join(cwd, 'temp')
    # print(temp_folder)
    if os.path.exists(temp_folder):
        try:
            shutil.rmtree(temp_folder)
        except:
            print("Can't remove temp folder")


def handle_temp_folder(cwd):
    if not cwd:
        cwd = os.getcwd()

    temp_folder = os.path.join(cwd, 'temp')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    return temp_folder


def match_modifiers(shortcut_modifiers, pressed_modifiers):
    if not shortcut_modifiers:
        return True

    for m in shortcut_modifiers:
        if m not in pressed_modifiers:
            return False

    return True


def read_min_max_stat(path_to_doverit):
    stat = {}
    with open(path_to_doverit, 'r') as f:
        for line in f:
            line = line.strip().split(';')
            stat[line[0]] = (float(line[3]), float(line[4]))

    return stat


def calc_areas(seg_results, lrm, verbose=False, cls_names=None, scale=1):
    if verbose:
        print(f"Старт вычисления площадей с lrm={lrm:0.3f}, всего {len(seg_results)} объектов:")

    areas = []
    for seg in seg_results:
        x_mass = seg.seg['x']
        y_mass = seg.seg['y']
        cls = seg.cls

        pol = []
        for x, y in zip(x_mass, y_mass):
            pol.append((x, y))

        pol = Polygon(pol)

        area = pol.area * lrm * lrm * scale * scale
        areas.append(area)

        if verbose:
            if cls_names:
                cls = cls_names[cls]
            print(f"\tплощадь {cls}: {area:0.3f} кв.м")

    return areas


def calc_point_mass_centroid(point_mass):
    p = Polygon(point_mass)
    c = p.centroid
    return [int(c.x), int(c.y)]


def get_extension(filename):
    return coords_calc.get_ext(filename)


def calc_width_proportions(percent):
    psnt_float = percent / 100.0

    monitors = screeninfo.get_monitors()
    min_width = 1e12
    for m in monitors:
        if m.width < min_width:
            min_width = m.width
    return int(min_width * psnt_float), int(min_width * (1.0 - psnt_float))


def convert_point_coords_to_geo(point_x, point_y, image_name, from_crs='epsg:3395', to_crs='epsg:4326'):
    img = Image.open(image_name)

    img_width, img_height = img.size

    geo_extent = coords_calc.get_geo_extent(image_name, from_crs=from_crs, to_crs=to_crs)

    lat_min, lat_max, lon_min, lon_max = geo_extent
    x = lon_min + (float(point_x) / img_width) * abs(lon_max - lon_min)
    y = lat_min + (1.0 - float(point_y) / img_height) * abs(lat_max - lat_min)

    return x, y


def is_dicts_equals(dict1, dict2):
    return all((dict1.get(k) == v for k, v in dict2.items()))


def get_label_colors(names, alpha=120):
    colors = {}
    if not alpha:
        alpha = 255

    for name in names:
        selected_color = config.COLORS[0]
        tek_color_num = 0
        is_break = False
        while selected_color in colors.values():
            tek_color_num += 1
            if tek_color_num == len(config.COLORS) - 1:
                is_break = True
                break
            selected_color = config.COLORS[tek_color_num]

        if is_break:
            selected_color = create_random_color(alpha)

        colors[name] = selected_color

    return colors


def read_yolo_yaml(yolo_yaml):
    with open(yolo_yaml, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_data


def convert_percent_to_alpha(percent, alpha_min=15, alpha_max=200):
    if percent:
        return alpha_min + int(percent * (alpha_max - alpha_min) / 100.0)

    return alpha_max


def set_alpha_to_max(rgba):
    return (rgba[0], rgba[1], rgba[2], 255)


def distance(p1, p2):
    return math.sqrt(pow(p1.x() - p2.x(), 2) + pow(p1.y() - p2.y(), 2))


def calc_abc(p1, p2):
    """
    Вычисление параметров A, B, C из уравнения прямой по двум заданным точкам p1 и p2
    """
    a = p2.y() - p1.y()
    b = -(p2.x() - p1.x())
    c = p2.x() * p1.y() - p2.y() * p1.x()
    return a, b, c


def distance_from_point_to_line(p, line_p1, line_p2):
    a, b, c = calc_abc(line_p1, line_p2)
    chisl = abs(a * p.x() + b * p.y() + c)
    znam = math.sqrt(a * a + b * b)
    if znam > 1e-8:
        return chisl / znam
    return 0


def distance_from_point_to_segment(point, seg_a_point, seg_b_point):
    A = point.x() - seg_a_point.x()
    B = point.y() - seg_a_point.y()
    C = seg_b_point.x() - seg_a_point.x()
    D = seg_b_point.y() - seg_a_point.y()

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        return distance(point, seg_a_point)

    if param > 1:
        return distance(point, seg_b_point)

    xx = seg_a_point.x() + param * C
    yy = seg_a_point.y() + param * D

    dx = point.x() - xx
    dy = point.y() - yy

    return math.sqrt(dx * dx + dy * dy)


def find_nearest_edge_of_polygon(polygon, point):
    d_min = 1e12
    edge = None
    size = len(polygon)
    for i in range(size):
        p1 = polygon[i]
        if i == size - 1:
            p2 = polygon[0]
        else:
            p2 = polygon[i + 1]
        if p1 != p2:
            d = distance_from_point_to_segment(point, p1, p2)  # distance_from_point_to_line(point, p1, p2)
            if d < d_min:
                d_min = d
                edge = p1, p2

    return edge


def calc_distance_to_nearest_edge(polygon, point):
    d_min = 1e12
    size = len(polygon)
    for i in range(size):
        p1 = polygon[i]
        if i == size - 1:
            p2 = polygon[0]
        else:
            p2 = polygon[i + 1]
        if p1 != p2:
            d = distance_from_point_to_segment(point, p1, p2)  # distance_from_point_to_line(point, p1, p2)
            if d < d_min:
                d_min = d

    return d_min


def get_closest_to_line_point(point, p1_line, p2_line):
    a, b, c = calc_abc(p1_line, p2_line)
    znam = a * a + b * b

    if znam != 0:
        x = (b * (b * point.x() - a * point.y()) - a * c) / znam

        y = (a * (-b * point.x() + a * point.y()) - b * c) / znam

        return QtCore.QPointF(x, y)

    return None


def create_random_color(alpha):
    rgba = [0, 0, 0, alpha]
    for i in range(3):
        rgba[i] = np.random.randint(0, 256)

    return rgba


def create_unique_image_name(image_name):
    splitted_name = image_name.split('.')
    new_name = ""
    for i in range(len(splitted_name) - 1):
        new_name += splitted_name[i]

    return f'{new_name} {datetime.datetime.now().microsecond}.{splitted_name[-1]}'


def calc_ellips_point_coords(ellipse_rect, angle):
    tl = ellipse_rect.topLeft()
    br = ellipse_rect.bottomRight()
    width = abs(br.x() - tl.x())
    height = abs(br.y() - tl.y())
    a = width / 2.0
    b = height / 2.0
    sn = math.sin(angle)
    cs = math.cos(angle)
    t = math.atan2(a * sn, b * cs)
    sn = math.sin(t)
    cs = math.cos(t)
    x_center = tl.x() + a
    y_center = tl.y() + b
    return QtCore.QPointF(x_center + a * cs, y_center + b * sn)


def convert_image_name_to_txt_name(image_name):
    splitted_name = image_name.split('.')
    txt_name = ""
    for i in range(len(splitted_name) - 1):
        txt_name += splitted_name[i]

    return txt_name + ".txt"


def sort_shapes_by_area(shapes, desc=True):
    """
    shapes - List({'points': List([x1,y1], [x2,y2], ...), 'cls_num' : Int, 'id': Int})
    desc - порядок. По умолчанию - по убыванию площади
    Нужно для нанесения сегментов на маску. Сперва большие сегменты, затем маленькие.
    Возвращает сортированный список shapes по площади
    """
    if desc:
        areas = [-Polygon(shape["points"]).area for shape in shapes]
    else:
        areas = [Polygon(shape["points"]).area for shape in shapes]
    idx = np.argsort(areas)
    return [shapes[i] for i in idx]


def paint_shape_to_mask(mask, points, cls_num):
    """
    Добавляет на mask - np.zeros((img_height, img_width))
    points:[ [x1,y1], ...] - точки граней маски
    cls_num - номер класса
    """
    img_height, img_width = mask.shape

    pol = Polygon(points)

    mask_image = features.rasterize([pol], out_shape=(img_height, img_width),
                                    fill=0,
                                    default_value=cls_num)

    mask[mask_image == cls_num] = cls_num


def filter_edged_points(points, width, height, tol=5):
    """
    points = List([x1,y1], [x2, y2] ... )
    """

    points_int = [[int(p[0]), int(p[1])] for p in points]
    points_new = []

    for p in points_int:
        x = p[0]
        y = p[1]
        if x > tol and x < width - tol and y > tol and y < height - tol:
            points_new.append([x, y])
    return points_new


def convert_text_name_to_image_name(text_name):
    splitted_name = text_name.split('.')
    img_name = ""
    for i in range(len(splitted_name) - 1):
        img_name += splitted_name[i]

    return img_name + ".txt"


def filter_masks(masks_results, conf_thres=0.2, iou_filter=0.3):
    """
    Фильтрация боксов
    conf_tresh - убираем все боксы с вероятностями ниже заданной
    iou_filter - убираем дубликаты боксов, дубликатами считаются те, значение IoU которых выше этого порога
    """
    unique_results = []
    skip_nums = []
    for i in range(len(masks_results)):
        if float(masks_results[i]['conf']) < conf_thres:
            continue

        biggest_mask = None

        if i in skip_nums:
            continue

        for j in range(i + 1, len(masks_results)):

            if j in skip_nums:
                continue

            if masks_results[i]['cls_num'] != masks_results[j]['cls_num']:
                continue

            if biggest_mask:
                pol1 = Polygon(biggest_mask['points'])
            else:
                pol1 = Polygon(masks_results[i]['points'])

            pol2 = Polygon(masks_results[j]['points'])

            un = pol1.union(pol2)
            inter = pol1.intersection(pol2)

            if inter and un:
                iou = inter.area / un.area

                if iou > iou_filter:

                    if pol1.area < pol2.area:
                        biggest_mask = masks_results[j]

                    skip_nums.append(j)

        if biggest_mask:
            unique_results.append(biggest_mask)
        else:
            unique_results.append(masks_results[i])

    return unique_results


def calc_area_by_points(points, lrm):
    pol = Polygon(points)

    return pol.area * lrm * lrm


def filter_results_by_areas(det_results, areas, stat):
    det_results_filtered = []
    for i, res in enumerate(det_results):

        cls_eng = ml_config.CLASSES_ENG[int(res.cls)]
        cls_dic = ml_config.CLASSES_RU[int(res.cls)]

        if cls_eng in stat:
            area = areas[i]
            if area < 0.9 * stat[cls_eng][0] or area > 1.1 * stat[cls_eng][1]:
                print(
                    f"\t\tплощадь {cls_dic} вне наблюдаемых границ +/- 10 %. Этот объект отфильтрован.")
                continue
            det_results_filtered.append(res)

    return det_results_filtered


if __name__ == '__main__':
    # img_name = 'F:\python\\ai_annotator\projects\\aes_big\\ano_google.jpg'
    # lrm = try_read_lrm(img_name)
    # print(lrm)
    # img = cv2.imread(img_name)
    # for i, part in enumerate(split_into_fragments(img, 450)):
    #     cv2.imshow(f'frag {i}', part)
    #     cv2.waitKey(0)

    points = [[100, 121], [200, 345], [0, 346]]

    print(filter_edged_points(points))
