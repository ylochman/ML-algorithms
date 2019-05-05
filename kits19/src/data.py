import torch
import torch.utils.data
import pandas as pd
import h5py


class H5CropData(torch.utils.data.Dataset):
    def __init__(self, filename, csv):
        self.filename = filename
        self.crops = pd.read_csv(csv)
        self.crops = self.crops[self.crops.kid_size > 0]
        # self.crops = self.crops[(self.crops.kid_size > 0) & (self.crops.case_id == 'case_00130')]

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        row = self.crops.iloc[idx]
        case_id = row.case_id
        z, x, y = eval(row.position)
        window = row.window
        self.file = h5py.File(self.filename, "r")
        dat = self.file[case_id][:, z:z + window, x:x + window, y:y + window]
        self.file.close()
        im = dat[0, :, :, :]
        mask = dat[1, :, :, :]
        return torch.from_numpy(im).unsqueeze(0).float(), torch.from_numpy(mask).long()

class H5CropData2(torch.utils.data.Dataset):
    def __init__(self, hdf5file, csvfile):
        self.filename = hdf5file
        crops = pd.read_csv(csvfile)
        non_empty = crops[crops.kid_size > 0]
        empty = crops[crops.kid_size == 0]
        empty = empty.sample(int(len(non_empty) * 0.2))
        self.crops = pd.concat([non_empty, empty])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        row = self.crops.iloc[idx]
        case_id = row.case_id
        z, y, x = eval(row.position)
        zw, yw, xw = eval(row.window_size)
        self.file = h5py.File(self.filename, "r")
        data = self.file[case_id][:, z:z+zw, y:y+yw, x:x+xw]
        self.file.close()
        im = data[0, :, :, :]
        mask = data[1, :, :, :]
        return torch.from_numpy(im).unsqueeze(0).float(), torch.from_numpy(mask).long()