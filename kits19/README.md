# KiTS 

## Structure
* src - code
* data - downloaded cases 
* starter_code - provided code

## V1 
Idea behind V1: Preprocess dataset and generate crops for each case. Because cases have are different depth and have big size, the following pipeline is applied:
* Pick window size =64
* Pad depth for each case to have `depth % window_size == 0`
* Generate positions for crops for each sample, with padding 32 and window size 64
* Save crop positions and meta information to csv, save padded original samples to hdf5 file.
### Preparing data
* Download data to data/ folder
* Execute `data_exploration` notebook and generate `data_stats.csv`
* Execute `h5_data_preparation` notebook and generate `crops.csv` and `crops.hdf5` file

### How to run training
* Activate env
* `python main_train.py --checkpoint unet.pth --epoches 5`


### Running tensorboard
```
docker run -it -p 9000:9000 -v $(pwd)/runs:/runs tensorflow/tensorflow /bin/bash

tensorboard --logdir=/runs/ --port=9000
```

## Port Forwarding
```
ssh -i ssh-keys/gpu-gc -L 9000:localhost:9000 sasha@34.74.74.127
```


# Screens
```
# Deattach screen
(ctrl-a-d) 
# Reattach screen
screen -r 
```



