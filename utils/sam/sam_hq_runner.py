import os

import cv2
import numpy as np

from utils.edges_from_mask import mask_to_polygons_layer
from utils.sam.segment_anything import sam_model_registry, SamPredictor


def mask2points(mask, simplify_factor=0.4):
    polygons = mask_to_polygons_layer(mask)

    results = []
    for pol in polygons:
        pol_simplified = pol.simplify(simplify_factor, preserve_topology=True)
        points = []
        try:
            xy = np.asarray(pol_simplified.boundary.xy, dtype="float")
            x_mass = xy[0].tolist()
            y_mass = xy[1].tolist()
            for x, y in zip(x_mass, y_mass):
                points.append([x, y])
            results.append(points)
        except:
            try:
                xy = np.asarray(pol_simplified.exterior.xy, dtype="float")
                x_mass = xy[0].tolist()
                y_mass = xy[1].tolist()
                for x, y in zip(x_mass, y_mass):
                    points.append([x, y])
                results.append(points)
            except:
                return []

    return results


def masks_to_shapes(masks, simplify_factor=0.1):
    shapes = []
    id_tek = 1

    for mask in masks:
        points_mass = mask2points(mask, simplify_factor=simplify_factor)

        for points in points_mass:
            cls_num = id_tek
            shape = {'id': id_tek, 'cls_num': cls_num, 'points': points}
            id_tek += 1
            shapes.append(shape)

    return shapes


class SAM_HQ:

    def __init__(self, model="vit_l", checkpoint="sam_hq_vit_l.pth", device='cpu'):
        """
        sam_weights - веса модели, 'sam_b.pt', 'sam_l.pt' или 'mobile_sam.pt'
        """
        sam = sam_model_registry[model](checkpoint=checkpoint)
        sam.to(device=device)
        self.device = device
        self.predictor = SamPredictor(sam)

    def set_image(self, source):
        # source - path to image or cv2.imread
        if os.path.isfile(source):
            source = cv2.imread(source)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(source)

    def everything(self):
        # Prepare a Prompt Process object

        print('SAM text prompt not implemented!')
        return []

    def box(self, bbox):
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        input_point, input_label = None, None

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=bbox,
            multimask_output=False,
            hq_token_only=False,
        )

        shapes = masks_to_shapes(masks)

        return shapes

    def text(self, text_prompt):
        # Text prompt
        print('SAM text prompt not implemented!')
        return []

    def point(self, points, pointlabel):
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        bbox = None
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=pointlabel,
            box=bbox,
            multimask_output=False,
            hq_token_only=False,
        )

        shapes = masks_to_shapes(masks)

        return shapes


if __name__ == '__main__':
    from time import process_time

    x1 = 400
    y1 = 200
    x2 = 800
    y2 = 600

    start = process_time()
    sam = SAM_HQ(device='cuda')
    print(f'Init time {process_time() - start}')

    start = process_time()
    sam.set_image("input_imgs/example0.png")
    print(f'Set image time {process_time() - start}')

    input_point = np.array([[x1, y1], [x2, y2]])
    input_label = np.array([1, 1])

    start = process_time()
    print(sam.point(input_point, input_label))
    print(f'Inference image time {process_time() - start}')
