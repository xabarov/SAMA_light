# Default path: in sama_light folder. Please change it for your SAM weight path

SAM_WEIGHT_PATH = {
    'SAM_base': 'weights/sam_b.pt',
    'SAM_large': 'weights/sam_l.pt',
    'FastSAM': 'weights/FastSAM-x.pt',
    'MobileSAM': 'weights/mobile_sam.pt',
    'EfficientSAM_small': 'weights/efficient_sam_vits.pt',
    'EfficientSAM_tiny': 'weights/efficient_sam_vitt.pt'

}

SAM_MODEL = 'FastSAM'  # 'EfficientSAM_small'

SAM_PLATFORM = 'cpu'  # 'cuda'
