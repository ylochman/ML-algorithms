import torch

config = {
    'CHECKPOINT': "unet.pth",
    'LR': 0.001,
    'L2': 0,
    'DEBUG': False,
    'CUDA': torch.cuda.is_available(),
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
