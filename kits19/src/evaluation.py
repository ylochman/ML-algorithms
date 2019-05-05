import torch
import torch.utils.data
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
import time
import src.starter.utils as starter
import src.starter.visualize as vis
import os
import h5py
import tqdm

from src.score import score_function_fast


class H5EvalCropData(torch.utils.data.Dataset):
    def __init__(self, filename, csv, case):
        self.filename = filename
        self.crops = pd.read_csv(csv)
        self.cases = self.crops.case_id.unique()
        self.case = case
        self.window = eval(self.crops[self.crops.case_id == case].iloc[0].window_size)

        self.positions = [eval(pos) for pos in self.crops[self.crops.case_id == case].position.values]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        z, x, y = self.positions[idx]
        zw, yw, xw = self.window
        self.file = h5py.File(self.filename, "r")
        dat = self.file[self.case][:, z:z + zw, x:x + xw, y:y + yw]
        self.file.close()
        im = dat[0, :, :, :]
        mask = dat[1, :, :, :]
        return torch.from_numpy(np.array([z, x, y])), \
               torch.from_numpy(im).unsqueeze(0).float(), \
               torch.from_numpy(mask).long()


def entries_count_mask(im_shape, crop_shape, positions):
    z, x, y = crop_shape
    mask = np.zeros(im_shape)
    for (zp, xp, yp) in positions:
        mask[zp:zp + z, xp:xp + x, yp:yp + y] += 1
    return mask


class Evaluator:
    def __init__(self, net, config, writer=None):
        net.eval()
        net.to(config['DEVICE'])
        self.device = config['DEVICE']
        self.net = net
        self.config = config
        self.scores = []
        self.global_step = 0
        self.epoch_number = 0
        if writer is None:
            self.tensorboard = SummaryWriter()
        else:
            self.tensorboard = writer

    def run(self, cases=None,
            crops_hdf_file="crops.hdf5",
            crops_csv_file="crops.csv",
            workers=0,
            batch_size=1,
            should_score=False,
            eval_file=None):
        if cases is None:
            crops = pd.read_csv(crops_csv_file)
            cases = crops.case_id.unique()

        self.scores.clear()
        with torch.no_grad():
            for case in tqdm.tqdm(cases):
                # Read ground truth mask
                file = h5py.File(crops_hdf_file, "r")
                gt_mask = file[case][1]
                file.close()
                # Create empty result mask
                result_mask = np.zeros((3, *gt_mask.shape))
                dataset = H5EvalCropData(crops_hdf_file, crops_csv_file, case)
                entries_mask = entries_count_mask(gt_mask.shape, dataset.window, dataset.positions)
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers)
                # Iterate over all crops and sum to the result mask
                for idx, (positions, image, target) in enumerate(tqdm.tqdm(loader, ascii=True)):
                    image, target = image.to(self.device), target.to(self.device)
                    predict = self.net(image)
                    z_window, x_window, y_window = dataset.window
                    for batch_item in range(positions.shape[0]):
                        z, x, y = positions[batch_item].numpy()
                        result_mask[:, z:z + z_window, x:x + x_window, y:y + y_window] += predict[batch_item].cpu().numpy()

                # Mean all prediction by crops
                result_mask = (result_mask / entries_mask)[-gt_mask.shape[0]:]
                if should_score:
                    tensor = torch.from_numpy(result_mask)
                    tensor = torch.softmax(tensor, 0)
                    tensor = torch.argmax(tensor, 0, keepdim=True)
                    predicted = tensor.numpy()
                    score = score_function_fast(predicted, gt_mask)
                    self.scores.append(score)
                    self.tensorboard.add_scalar("val_score", score, global_step=self.global_step)

                if eval_file is not None:
                    pred_file = h5py.File(eval_file, "a")
                    pred_file.create_dataset(case, data=result_mask)
                    pred_file.close()
                self.global_step += 1
            self.tensorboard.add_scalar("val_epoch_score", np.mean(self.scores), global_step=self.epoch_number)
            self.epoch_number += 1
