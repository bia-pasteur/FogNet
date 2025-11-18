# FogNet: Flow-Guided U-Net
## Biological Object Segmentation on Video data using Optical Flow-Guided Temporal Consistency

### Installation guide

Currently, FogNet is not hosted on PiPy, so you must first clone the repo, then inside the repository, run
    
```bash
$ pip install .
```
All depencencies should be installed automatically.

### Getting started

#### Data folders 

The current data loading pipeline (SegmentationDatamodule) in order to run from CLI as intented expect the data in the following format:

root_data_folder/
├── train/
│    ├── masks
│    |   ├── 0000.tiff
│    |   └── 0001.tiff
│    |   └── ...
|    ├── video.tiff
├── val/
│    ├── masks
│    |   ├── 0000.tiff
│    |   └── 0001.tiff
│    |   └── ...
|    ├── video.tiff
├── test/
│    ├── masks
│    |   ├── 0000.tiff
│    |   └── 0001.tiff
│    |   └── ...
|    ├── video.tiff



#### Running from CLI

```bash
$ python train.py fit --config ./config/config.yaml
```
