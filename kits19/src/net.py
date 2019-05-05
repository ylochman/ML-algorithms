from src.unet3 import UNet3D
from src.unet3_dsv import unet_grid_attention_3D

from src.unet3_dsv import unet_CT_multi_att_dsv_3D
from src.utils import count_parameters


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
