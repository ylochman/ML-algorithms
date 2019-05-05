from tqdm import tqdm
import h5py
import src.starter.utils as starter
import pandas as pd
import numpy as np
import argparse

DEFAULT_WIN_D = 64
DEFAULT_WIN_H = 256
DEFAULT_WIN_W = 256

DEFAULT_STRIDE_D = 32
DEFAULT_STRIDE_H = 128
DEFAULT_STRIDE_W = 128

def normalize(volume, min_val, max_val):
    im_volume = (volume - min_val) / max(max_val - min_val, 1e-3)
    return im_volume

def positions(im_shape, crop_shape, strides):
        zi, xi, yi = im_shape
        zc, xc, yc = crop_shape
        zs, xs, ys = strides
        for z in [z for z in range(0, zi, zs) if z + zc <= zi]:
            for x in [x for x in range(0, xi, xs) if x + xc <= xi]:
                for y in [y for y in range(0, yi, ys) if y + yc <= yi]:
                    yield (z, x, y)
                    
def get_pad_size(imsize, cropsize):
    if imsize % cropsize == 0:
        return 0
    else:
        return cropsize - imsize % cropsize
    
def halved_pads(pad_size):
    return (pad_size//2, pad_size//2 + pad_size%2)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--stage", help="Stage (train/val)", type=str, default="train")
    parser.add_argument("--win_w", help="Window width (and height)", type=int, default=DEFAULT_WIN_W)
    parser.add_argument("--win_d", help="Window depth", type=int, default=DEFAULT_WIN_D)
    parser.add_argument("--stride_w", help="Stride width (and height)", type=int, default=DEFAULT_STRIDE_W)
    parser.add_argument("--stride_d", help="Stride depth", type=int, default=DEFAULT_STRIDE_D)
    
    args = parser.parse_args()
    
    D_SIZE = args.win_d
    H_SIZE = args.win_w
    W_SIZE = args.win_w
    WINDOW = (D_SIZE, H_SIZE, W_SIZE)
    
    D_STRIDE = args.stride_d
    H_STRIDE = args.stride_w
    W_STRIDE = args.stride_w
    STRIDE = (D_STRIDE, H_STRIDE, W_STRIDE)
    
    data = pd.read_csv("data_interpolated_stats.csv")
    max_val = data['max_val'].quantile(0.95)
    min_val = data['min_val'].quantile(0.01)
    
    data = pd.read_csv("{}_data_interpolated_stats.csv".format(args.stage))
    cropsfile = h5py.File("data_ready/{}_{}_{}_{}.hdf5".format(args.stage, H_SIZE, W_SIZE, D_SIZE), "w")
    cropscsv = "data_ready/{}_{}_{}_{}.csv".format(args.stage, H_SIZE, W_SIZE, D_SIZE)
    
    crops_data = []
    for i in tqdm(range(len(data))):
        row = data.iloc[i]
        case_id, z, x, y = row['case_id'], row['num_slices'], row['height'], row['width']
        im, mask = starter.load_case(case_id)
        im, mask = im.get_data(), mask.get_data()
    #     print("Before padding:", im.shape)

        z_pad_size = get_pad_size(im.shape[0], D_SIZE)
        y_pad_size = get_pad_size(im.shape[1], H_SIZE)
        x_pad_size = get_pad_size(im.shape[2], W_SIZE)

        im = normalize(im, min_val, max_val)
        im = np.pad(im, (halved_pads(z_pad_size),
                         halved_pads(y_pad_size),
                         halved_pads(x_pad_size)), 'constant')
        mask = np.pad(mask, (halved_pads(z_pad_size),
                             halved_pads(y_pad_size),
                             halved_pads(x_pad_size)), 'constant')
    #     print("After padding:", im.shape)
        for position in positions((int(z + z_pad_size), int(y + y_pad_size), int(x + x_pad_size)),
                                  WINDOW, STRIDE):
            z, y, x = position
            cropped_mask = mask[z:z+D_SIZE, y:y+H_SIZE, x:x+W_SIZE]
            kid_size = np.sum(cropped_mask == 1)
            tumor_size = np.sum(cropped_mask == 2)
            crops_data.append([case_id, position, WINDOW, STRIDE, kid_size, tumor_size])
        cropsfile.create_dataset(case_id, data=np.array([im, mask]))

    cropsfile.close()

    crops_data = pd.DataFrame(data=crops_data,
                              columns=["case_id", "position", "window_size",
                                       "stride", "kid_size", "tumor_size"])
    crops_data.to_csv(cropscsv)

if __name__ == "__main__":
    main()