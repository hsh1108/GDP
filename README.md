# Geomagnetic Disturbance Prediction Project
This is a project for Geomagnetic Disturbance Prediction(GDP) with python 3.6 and tensorflow.

## Dataset
RRA(National Radio Research Agency) provides two kinds of processed data. 
- The solar wind data observed by ACE satellites of NASA Goddard Space Flight Center.
- The average value of the geomagnetic disturbance index measured at 8 stations around the world.

This project aims to predict geomagnetic disturbance index using solar wind data.
Refer to the following site: https://sapiensteam.com/make/contestDetail.do 

## How to train
Download and prepare a dataset.
You can also get dataset manually at the following site: https://sapiensteam.com/make/contestDetail.do
```bash
$ data/download_data.sh
```

Train model.
```bash
$ python train.py
```

## How to test
Test model.
```bash
$ python test.py
```

## Result
Not yet..

