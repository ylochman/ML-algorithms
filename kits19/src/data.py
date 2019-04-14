import torch
import torch.utils.data
import pandas as pd
import h5py


class H5CropData(torch.utils.data.Dataset):
    def __init__(self, filename, csv):
        self.crops = pd.read_csv(csv)
        self.file = h5py.File(filename, "r")

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        row = self.crops.iloc[idx]
        case_id = row.case_id
        z, x, y = eval(row.position)
        window = row.window
        dat = self.file[case_id][:, z:z + window, x:x + window, y:y + window]
        im = dat[0, :, :, :]
        mask = dat[1, :, :, :]
        return torch.from_numpy(im).unsqueeze(0).float(), torch.from_numpy(mask).long()
