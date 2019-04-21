import torch.utils.data

from src.config import config
from src.data import H5CropData
from src.evaluation import Evaluator
from src.train import Trainer
from src.unet import UNet3D
from src.utils import load_checkpoint
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch", help="Number of epochs to train", type=int, default=1)
parser.add_argument("--workers", help="Checkpoint name", type=int, default=6)

parser.add_argument("--checkpoint", help="Checkpoint name", type=str, default=None)
parser.add_argument("--score", help="Checkpoint name", type=bool, default=False)

args = parser.parse_args()
print("Arguments: {}".format(args))
print("Config: {}".format(config))

net = UNet3D(1, 3, False)
if args.checkpoint is not None:
    extra = load_checkpoint(net, args.checkpoint)

crops = pd.read_csv("crops.csv")
cases = crops.case_id.unique()

evaluator = Evaluator(net, config)
evaluator.run(["case_00004"], workers=args.workers, batch_size=args.batch, score=args.score)



