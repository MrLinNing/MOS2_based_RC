# Vessel segmentation and extraction




## Data download
The original data file `DRIVE` can be downloaded from  [LadderNet](https://github.com/juntang-zhuang/LadderNet). 


## MOS2 Data file is shown as `Slope_of_64.xlsx`


## Data process 


### To generate hdf5 file of training data
```
python ori_prepare_datasets_DRIVE.py  ## original image
python prepare_datasets_DRIVE.py   ## quant image 

```


## HyperParameter defination
```
ori_configuration.txt  ## original image define
configuration.txt      ## MOS2 based quant image define
```


## Run simulations for original image

```
cd src
python retinaNN_training_ori.py
python retinaNN_predict_ori.py
```


## Run simulations for MOS2 based quant image

```
cd src
python retinaNN_training.py
python retinaNN_predict.py
```

Then, the simulation results are in `test` and `test_ori` files


## References

Most of the code in this simulation is based on an open GitHub repository. We would like to extend our gratitude to the original author for their contributions and efforts. You can find the original repository at the following link:

[LadderNet](https://github.com/juntang-zhuang/LadderNet)
