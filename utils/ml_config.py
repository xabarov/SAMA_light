# Default path: in sama_light folder. Please change it for your SAM weight path

SAM_WEIGHT_PATH = {
    'SAM_base': 'weights/sam_b.pt',
    'SAM_large': 'weights/sam_l.pt',
    'FastSAM': 'weights/FastSAM-x.pt',
    'MobileSAM': 'weights/mobile_sam.pt',
    'EfficientSAM_small': 'weights/efficient_sam_vits.pt',
    'EfficientSAM_tiny': 'weights/efficient_sam_vitt.pt',
    'SAM_HQ_B': 'weights/sam_hq_vit_b.pth',
    'SAM_HQ_L': 'weights/sam_hq_vit_l.pth',
    'SAM_HQ_H': 'weights/sam_hq_vit_h.pth',
    'SAM_HQ_TINY': 'weights/sam_hq_vit_tiny.pth'
}

SAM_MODEL = 'FastSAM'  # 'EfficientSAM_small'

SAM_PLATFORM = 'cpu'  # 'cuda'
