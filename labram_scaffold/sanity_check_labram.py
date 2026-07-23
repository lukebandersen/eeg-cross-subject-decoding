import sys
sys.path.append(r"C:\Users\mwolff3\Desktop\EEG_Image_decode-develop\LaBraM")

import torch
from models import LaBraMBaseEncoder, get_input_chans

CKPT_PATH = r"C:\Users\mwolff3\Desktop\EEG_Image_decode-develop\LaBraM\checkpoints\labram-base.pth"

CHANNELS = [
    'Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz',
    'P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8',
    'FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7',
    'FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8',
    'P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4','AF8',
]

enc = LaBraMBaseEncoder(patch_size=200, embed_dim=200, depth=12, num_heads=10,
                        pretrained_ckpt=CKPT_PATH, pooling="mean")
print("backend:", enc.backend)

input_chans = get_input_chans(CHANNELS)

x = torch.randn(2, len(CHANNELS), 1, 200)
feats = enc(x, input_chans)
print("output shape:", feats.shape)
