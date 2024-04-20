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

#### FastSAM needs YOLOv8 FastSAM weight 'FastSAM-x.pt', so you have 2 options:
1. Download weight manually from: https://disk.yandex.ru/d/47FN8Wh8dnoJpQ and put it to `sama_light` dir
2. Or start `annotator_sam.py` and ultralytics downloads it for you automatically




