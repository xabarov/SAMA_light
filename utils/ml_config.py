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
    'SAM_HQ_TINY': 'weights/sam_hq_vit_tiny.pth',
    'SAM2_large': 'weights/sam2_l.pt',
    'SAM2_base': 'weights/sam2_b.pt',
    'SAM2_small': 'weights/sam2_s.pt',
    'SAM2_tiny': 'weights/sam2_t.pt',
}

SAM_MODEL = 'SAM2_large'

SAM_PLATFORM = 'cuda'  # or 'cpu'
