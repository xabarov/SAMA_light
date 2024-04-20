import os

from ultralytics import FastSAM
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam.predict import DEFAULT_CFG
from utils.sam.fast_sam_prompt import FastSAMPrompt
from utils.edges_from_mask import mask_to_polygons_layer
import numpy as np


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
            pass

    return results


def ann_to_shapes(ann, simplify_factor=0.1):
    shapes = []
    id_tek = 1
    for res in ann:
        mask = res.cpu().numpy()
        mask[mask == 1] = 255
        points_mass = mask2points(mask, simplify_factor=simplify_factor)

        for points in points_mass:
            cls_num = id_tek
            shape = {'id': id_tek, 'cls_num': cls_num, 'points': points}
            id_tek += 1
            shapes.append(shape)

    return shapes


def results_to_ann(results):
    return results[0].masks.data


class FastSAMCustom:

    def __init__(self, sam_weights='FastSAM-x.pt'):
        """
        sam_weights - веса модели, 'FastSAM-s.pt' или 'FastSAM-x.pt'
        """

        # Create a FastSAM model
        self.model = FastSAM(sam_weights)

    def prepare_prompt_process(self, source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9):
        # Run inference on an image
        self.image_everything_results = self.model(source, device=device, retina_masks=retina_masks, imgsz=imgsz,
                                                   conf=conf,
                                                   iou=iou)

        self.prompt_process = FastSAMPrompt(source, self.image_everything_results, device=device)

    def everything(self):
        # Prepare a Prompt Process object

        # Everything prompt
        ann = self.prompt_process.everything_prompt()
        return ann_to_shapes(ann)

    def box(self, bbox):
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = self.prompt_process.box_prompt(bbox=bbox)
        return ann_to_shapes(ann)

    def text(self, text_prompt):
        # Text prompt
        ann = self.prompt_process.text_prompt(text=text_prompt)
        return ann_to_shapes(ann)

    def point(self, points, pointlabel):
        # Point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        ann = self.prompt_process.point_prompt(points=points, pointlabel=pointlabel)

        shapes = ann_to_shapes(ann)

        return shapes


class SAMCustom:

    def __init__(self, sam_weights='sam_b.pt', device='cpu', imgsz=1024,
                 conf=0.25):
        """
        sam_weights - веса модели, 'sam_b.pt', 'sam_l.pt' или 'mobile_sam.pt'
        """
        overrides = dict(conf=conf, task='segment', mode='predict', imgsz=imgsz, model=sam_weights, device=device,
                         save=False)
        cfg = DEFAULT_CFG
        cfg.save_dir = os.path.join(os.path.dirname(__file__), 'sam_results')
        self.predictor = SAMPredictor(cfg=cfg, overrides=overrides)

    def set_image(self, source):
        # source - path to image or cv2.imread
        self.predictor.set_image(source)

    def everything(self, crop_n_layers=1, points_stride=12):
        # Prepare a Prompt Process object

        # Everything prompt
        results = self.predictor(crop_n_layers=crop_n_layers, points_stride=points_stride)
        ann = results_to_ann(results)
        return ann_to_shapes(ann)

    def box(self, bbox):
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        results = self.predictor(bboxes=bbox)
        ann = results_to_ann(results)
        return ann_to_shapes(ann)

    def text(self, text_prompt):
        # Text prompt
        print('SAM text prompt not implemented!')
        return []

    def point(self, points, pointlabel):
        # Point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        results = self.predictor(points=points, labels=pointlabel)
        ann = results_to_ann(results)
        shapes = ann_to_shapes(ann)

        return shapes


def test_sam(source='bus.png',
             bbox=[200, 200, 300, 300],
             points=[[200, 200]],
             pointlabel=[1]):
    start = process_time()
    ys = SAMCustom(device='cuda')
    ys.set_image(source)
    print(f"Init time: {process_time() - start}")

    start = process_time()
    ann = ys.everything()
    print(f"Everything time: {process_time() - start}")

    start = process_time()
    shapes = ys.box(bbox)

    print(f"Bbox time: {process_time() - start}")

    start = process_time()
    shapes = ys.point(points, pointlabel)
    print(shapes)
    print(f"Point time: {process_time() - start}")


def test_fast_sam(source='bus.png',
                  bbox=[200, 200, 300, 300],
                  points=[[200, 200]],
                  pointlabel=[1],
                  text_prompt='a photo of a dog'):
    start = process_time()
    ys = FastSAMCustom()
    ys.prepare_prompt_process(source, device='cuda')
    print(f"Init time: {process_time() - start}")

    start = process_time()
    ann = ys.everything()
    print(f"Everything time: {process_time() - start}")

    start = process_time()
    shapes = ys.box(bbox)

    print(f"Bbox time: {process_time() - start}")

    start = process_time()
    shapes = ys.text(text_prompt)
    print(shapes)
    print(f"Text time: {process_time() - start}")

    start = process_time()
    shapes = ys.point(points, pointlabel)
    print(shapes)
    print(f"Point time: {process_time() - start}")


if __name__ == '__main__':
    from time import process_time

    test_sam()
