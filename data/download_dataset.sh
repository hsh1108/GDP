#!/usr/bin/env bash

# Geomagnetic Disturbance dataset.
if [ ! -r data/dataset ]
then
    echo "Downloading Geomagnetic Disturbance dataset."
    mkdir -p data/dataset
    pushd data/dataset
    wget --progress=bar \
        -r 'https://docs.google.com/uc?export=download&id=12r-BtDGb_8L0z5bHpyP6xwBx-r2JRp1-' \
        -O dataset.zip
    unzip dataset.zip
    rm dataset.zip
    popd
fi
      
     


