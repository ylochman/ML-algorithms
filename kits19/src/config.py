import torch

config = {
    'CHECKPOINT': "unet.pth",
    'LR': 0.001,
    'L2': 0,
    'DEBUG': False,
    'CUDA': torch.cuda.is_available(),
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'WINDOW_SIZE': (32, 128, 128),
    'STRIDE': (16, 64, 64)
}
