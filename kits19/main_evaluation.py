import torch.utils.data

from src.config import config
from src.data import H5CropData
from src.evaluation import Evaluator
from src.train import Trainer
from src.unet import UNet3D
from src.unet_dsv import unet_CT_multi_att_dsv_3D
from src.utils import load_checkpoint
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--evalbatch", help="Number of epochs to train", type=int, default=1)
parser.add_argument("--workers", help="Checkpoint name", type=int, default=6)
parser.add_argument("--net", help="Neural network", type=str, default="3dunet")
parser.add_argument("--checkpoint", help="Checkpoint name", type=str, default=None)
parser.add_argument("--score", help="Checkpoint name", type=bool, default=True)

args = parser.parse_args()
print("Arguments: {}".format(args))
print("Config: {}".format(config))

if args.net == '3dunet':
    net = UNet3D(1, 3, False)
elif args.net == '3ddsv':
    net = unet_CT_multi_att_dsv_3D(n_classes=3, in_channels=1)
else:
    raise NotImplementedError()

if args.checkpoint is not None:
    extra = load_checkpoint(net, args.checkpoint)

evaluator = Evaluator(net, config)

evaluator.run(crops_csv_file="val_interpolated_crops.csv", crops_hdf_file="val_interpolated_crops.hdf5",
              workers=args.workers, batch_size=args.evalbatch, should_score=args.score,
              eval_file="val_predictions.hdf5")
