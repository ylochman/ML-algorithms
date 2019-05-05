import argparse

from src.config import config
from src.evaluation import Evaluator
from src.net import build_network
from src.utils import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--evalbatch", help="Number of epochs to train", type=int, default=1)
parser.add_argument("--workers", help="Checkpoint name", type=int, default=6)
parser.add_argument("--net", help="Neural network", type=str, default="3dunet")
parser.add_argument("--checkpoint", help="Checkpoint name", type=str, default=None)
parser.add_argument("--score", help="Checkpoint name", type=bool, default=True)
parser.add_argument("--file", help="File name", type=str, default="val_predictions.hdf5")

args = parser.parse_args()
print("Arguments: {}".format(args))
print("Config: {}".format(config))

net = build_network(args.net)
if args.checkpoint is not None:
    extra = load_checkpoint(net, args.checkpoint)

evaluator = Evaluator(net, config)

evaluator.run(crops_csv_file="val_interpolated_crops.csv", crops_hdf_file="val_interpolated_crops.hdf5",
              workers=args.workers, batch_size=args.evalbatch, should_score=args.score,
              eval_file=args.file)
