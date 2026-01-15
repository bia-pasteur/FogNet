
🌫️ FogNet: Flow-Guided U-Net
=============================

### Biological Object Segmentation in Video Using Optical-Flow-Guided Temporal Consistency

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.x-purple)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Code-orange)

FogNet is a deep learning framework for biological object segmentation in video data.  
It extends a U-Net architecture with **optical-flow-guided temporal consistency**, enabling robust segmentation across time.

* * *

📦 Installation
---------------

FogNet is not currently hosted on PyPI. To install it locally:

```bash
git clone <repo-url>
cd FogNet
pip install .
```

All required dependencies should be installed automatically.

> **Reproducibility note**  
> If you encounter dependency issues, the file `req_reproducibility.txt` contains the **exact package versions used for the paper**.

* * *

🚀 Getting Started
------------------

### 📁 Data Folder Structure

The current data loading pipeline (`SegmentationDataModule`) is designed to run from the CLI and expects the following directory structure by default:

```
root_data_folder/
├── train/
│   ├── masks/
│   │   ├── 0000.tiff
│   │   ├── 0001.tiff
│   │   └── ...
│   └── video.tiff
├── val/
│   ├── masks/
│   │   ├── 0000.tiff
│   │   ├── 0001.tiff
│   │   └── ...
│   └── video.tiff
├── test/
│   ├── masks/
│   │   ├── 0000.tiff
│   │   ├── 0001.tiff
│   │   └── ...
│   └── video.tiff
```

#### Custom dataset layouts

All folder and subfolder names can be customized via the configuration files.

For the **SINETRA dataset**, the default settings are:

*   `subfolders: ["111", "222", "333"]` instead of `["train", "val", "test"]`
*   `mask_dir_name: "tracks"` instead of `"masks"`

* * *

### 🖥️ Running from the Command Line

Training is launched via PyTorch Lightning’s CLI:

```bash
python train.py fit --config ./config/config.yaml
```

By default:

*   If **validation and test sets** are provided:
    *   The decision threshold is optimized on the validation set.
    *   Final performance metrics are computed on the test set.
*   Model checkpoints, evaluation results, and TensorBoard logs are saved in the training logs directory.
*   The model name and version can be customized from the configuration file.

* * *

📝 Remarks & Training Details
-----------------------------

*   All model, data, and training parameters are defined in `config/config.yaml` when running from the CLI.
*   Default settings:
    *   **Window size**: 3 (fastest configuration)
    *   **Training epochs**: 30
*   The training process saves:
    *   The **last checkpoint**
    *   The **best checkpoint**, selected based on the validation loss

### Optical Flow Computation

*   If `precalculate_flow: true` is set in both the **model** and **data** configurations:
    *   Optical flow is computed **once before training**
    *   This significantly speeds up training
    *   Requires additional RAM to store flow values for the full dataset

### Dataset Splitting

*   If only a training set is provided and `train_val_test_split: true`:
    *   The dataset is automatically split into:
        *   80% training
        *   10% validation
        *   10% test

### Performance Controls

*   To accelerate experimentation, you can limit the dataset size:
    *   Default maximum samples:
        *   Train: 200
        *   Validation: 50
        *   Test: 50
