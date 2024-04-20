# SAMA Light Annotator

Labeling Images for Object Detection and Instance Segmentation

![alt text](assets/demo.gif)

## Install

### For light version without SAM run:

1. `pip install -r requirements_light.txt`
2. `python annotator_light.py`

### For version with SAM run:

1. `pip install -r requirements_sam.txt`
2. `python annotator_sam.py`

### SAM settings

1. Set `SAM_WEIGHT_PATH`, `SAM_MODEL` and `SAM_PLATFORM` in `utils.ml_config.py`. By default SAMA Light use **FastSAM**
   with 'FastSAM-x.pt' on 'cpu'
2. Download weight manually and place it according `ml_config.py SAM_WEIGHT_PATH` variable or start `annotator_sam.py`
   and [ultralytics](https://docs.ultralytics.com/models/fast-sam/) downloads it for you automatically. 
3. Default path `FastSAM-x.pt` in config suggests that weights are
   in ***sama_light/*** project folder. You can download it from:
- [FastSAM](https://docs.ultralytics.com/models/fast-sam/)
- [SAM](https://docs.ultralytics.com/models/fast-sam/)
- [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/)




