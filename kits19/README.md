# Kidney Tumor Segmentation Challenge 2019

See our [project report](./report/final-paper.pdf)

## Project structure
* `./src` - source code
* `./src/starter` - provided starter code from [2019 KiTS Challenge Repository](https://github.com/neheller/kits19)
* `./data` - cases downloaded from [kits19/master/data](https://github.com/neheller/kits19/tree/master/data)
* `./data_interpolated` - cases downloaded from [kits19/interpolated/data](https://github.com/neheller/kits19/tree/interpolated/data)
* `./report` - project presentations and report
* `./*.ipynb` - various research notebooks
* `./*.csv` - various data statistics files
* Pipeline files:
	* `./prepare_data.py`
	* `./pipeline.py`
	* `./main_evaluation.py`
	* `./main_train.py`

## To reproduce results..

### Prepare data
* [Download](https://github.com/neheller/kits19/tree/interpolated/data) data to `./data_interpolated` (*Old: [Download](https://github.com/neheller/kits19/tree/master/data) data to `./data`*)
* Execute `data_exploration` notebook -- generate `data_stats.csv`
* Execute `h5_data_preparation` notebook -- generate `crops.csv` and `crops.hdf5` file

### Prepare an environment
```bash
virtualenv --python=python3 .env
source .env/bin/activate
pip install requirements.txt
```

### Run training
```bash
python main_train.py --checkpoint unet.pth
```

## Technical Details

### Run Tensorboard
```bash
docker run -it -p 9000:9000 -v $(pwd)/runs:/runs tensorflow/tensorflow /bin/bash
tensorboard --logdir=/runs/ --port=9000
```
or run this command from `./project`:
```bash
docker run -d -p 9000:9000 -v $(pwd)/runs:/runs tensorflow/tensorflow /bin/bash -c "tensorboard --logdir=/runs/ --port=9000"
```

### Port forwarding
```bash
ssh -i ssh-keys/gpu-gc -L 9000:localhost:9000 sasha@34.74.74.127
```

### Use Screen
```bash
# Deattach screen
(ctrl-a-d) 
# Reattach screen
screen -r 
```
