import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os

from utils.sam.efficient_sam.build_efficient_sam import build_efficient_sam

from utils.edges_from_mask import mask_to_polygons_layer


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


def mask_to_shapes(mask, simplify_factor=0.1):
    shapes = []
    id_tek = 1
    points_mass = mask2points(mask, simplify_factor=simplify_factor)

    for points in points_mass:
        cls_num = id_tek
        shape = {'id': id_tek, 'cls_num': cls_num, 'points': points}
        id_tek += 1
        shapes.append(shape)

    return shapes


class EfficientSAM:

    def __init__(self, model='small', checkpoint="weights/efficient_sam_vits.pt", device='cpu'):
        """
        sam_weights - веса модели, 'sam_b.pt', 'sam_l.pt' или 'mobile_sam.pt'
        """
        self.device = device
        if model == 'small':
            self.model = self.build_efficient_sam_vits(checkpoint).to(device)
        else:
            self.model = self.build_efficient_sam_vitt(checkpoint).to(device)

    def build_efficient_sam_vitt(self, checkpoint="weights/efficient_sam_vitt.pt"):
        return build_efficient_sam(
            encoder_patch_embed_dim=192,
            encoder_num_heads=3,
            checkpoint=checkpoint,
        ).eval()

    def build_efficient_sam_vits(self, checkpoint="weights/efficient_sam_vits.pt"):
        return build_efficient_sam(
            encoder_patch_embed_dim=384,
            encoder_num_heads=6,
            checkpoint=checkpoint,
        ).eval()

    def set_image(self, source):
        # source - path to image or cv2.imread
        if os.path.isfile(source):
            source = np.array(Image.open(source))
        self.source = transforms.ToTensor()(source).to(self.device)

    def everything(self):
        # Prepare a Prompt Process object

        print('SAM text prompt not implemented!')
        return []

    def box(self, bbox):
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        pts_sampled = torch.reshape(torch.tensor([[bbox[0], bbox[1]], [bbox[2], bbox[3]]]), [1, 1, -1, 2]).to(self.device)
        pts_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, -1]).to(self.device)
        predicted_logits, predicted_iou = self.model(
            self.source[None, ...],
            pts_sampled,
            pts_labels,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        # The masks are already sorted by their predicted IOUs.
        # The first dimension is the batch size (we have a single image. so it is 1).
        # The second dimension is the number of masks we want to generate (in this case, it is only 1)
        # The third dimension is the number of candidate masks output by the model.
        # For this demo we use the first mask.
        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()

        shapes = mask_to_shapes(mask.astype(int) * 255)

        return shapes

    def text(self, text_prompt):
        # Text prompt
        print('SAM text prompt not implemented!')
        return []

    def point(self, points, pointlabel):
        # Point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        pts_sampled = torch.reshape(torch.tensor(points), [1, 1, -1, 2]).to(self.device)
        pts_labels = torch.reshape(torch.tensor(pointlabel), [1, 1, -1]).to(self.device)
        predicted_logits, predicted_iou = self.model(
            self.source[None, ...],
            pts_sampled,
            pts_labels,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        # The masks are already sorted by their predicted IOUs.
        # The first dimension is the batch size (we have a single image. so it is 1).
        # The second dimension is the number of masks we want to generate (in this case, it is only 1)
        # The third dimension is the number of candidate masks output by the model.
        # For this demo we use the first mask.
        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
        return mask_to_shapes(mask.astype(int) * 255)


if __name__ == '__main__':
    from time import process_time

    x1 = 400
    y1 = 200
    x2 = 800
    y2 = 600

    start = process_time()
    sam = EfficientSAM(device='cuda')
    print(f'Init time {process_time()-start}')

    start = process_time()
    sam.set_image("figs/examples/dogs.jpg")
    print(f'Set image time {process_time() - start}')

    input_point = np.array([x1, y1, x2, y2])
    input_label = np.array([1, 1])

    start = process_time()
    print(sam.box(input_point))
    print(f'Inference image time {process_time() - start}')