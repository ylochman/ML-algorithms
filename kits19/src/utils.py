import os
import h5py
import torch

from src.unet3 import UNet3D
from src.unet3_dsv import unet_grid_attention_3D

from src.unet3_dsv import unet_CT_multi_att_dsv_3D

CHECKPOINT_DIR = ""


def load_checkpoint(model, checkpoint, optimizer=None):
    exists = os.path.isfile(CHECKPOINT_DIR + checkpoint)
    if exists:
        state = torch.load(CHECKPOINT_DIR + checkpoint)
        model.load_state_dict(state['state_dict'], strict=False)
        optimizer_state = state.get('optimizer')
        if optimizer and optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        print("Checkpoint loaded: %s " % state['extra'])
        return state['extra']
    else:
        print("Checkpoint not found")
    return {'epoch': 0, 'lb_acc': 0}


def save_checkpoint(model, extra, checkpoint, optimizer=None):
    state = {'state_dict': model.state_dict(),
             'extra': extra}
    if optimizer:
        state['optimizer'] = optimizer.state_dict()

    torch.save(state, CHECKPOINT_DIR + checkpoint)
    print('model saved to %s' % (CHECKPOINT_DIR + checkpoint))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_network(net_name):
    if net_name == '3dunet':
        net = UNet3D(1, 3, False)
    elif net_name == 'grid':
        net = unet_grid_attention_3D(n_classes=3, in_channels=1)
    elif net_name == 'dsv':
        net = unet_CT_multi_att_dsv_3D(n_classes=3, in_channels=1)
    else:
        raise NotImplementedError()
    print("Network {} created. Parameters: {}".format(net_name, count_parameters(net)))
    return net
