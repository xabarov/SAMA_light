# SAMA Light Annotator

Labeling Images for Object Detection, Instance Segmentation and Semantic Segmentation

![alt text](assets/demo.gif)

## Install

### For light version without SAM run:

1. `pip install -r requirements_light.txt`
2. `python annotator_light.py`

### For version with SAM run:

1. `pip install -r requirements_sam.txt`
2. `python annotator_sam.py`

### SAM settings

1. Set `SAM_WEIGHT_PATH`, `SAM_MODEL` and `SAM_PLATFORM` in `utils.ml_config.py`. By default SAMA Light use **SAM2_large** on 'cuda'
2. List of supported methods: SAM, SAM-HQ, SAM-2, MobileSAM, EfficientSAM, FastSAM
3. For models SAM2, FastSAM weights will be downloaded automatically.
4. For other models you have to download weight manually and place it according `ml_config.py SAM_WEIGHT_PATH` variable. You can download weights from:
- [SAM_HQ](https://github.com/SysCV/sam-hq)
- [FastSAM](https://docs.ultralytics.com/models/fast-sam/)
- [SAM_2](https://docs.ultralytics.com/models/sam-2)
- [SAM](https://docs.ultralytics.com/models/fast-sam/)
- [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/)
- [EfficientSAM](https://github.com/yformer/EfficientSAM/tree/main)
