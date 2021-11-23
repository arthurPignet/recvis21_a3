## MVA Object recognition and computer vision 2021/2022

### Assignment 3: Image classification 

**Author**: Arthur Pignet

#### Original challenge

This repository is based on the assigment's original repository, which can be found here: https://github.com/willowsierra/recvis21_a3.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── Report_A3_Arthur_Pignet.pdf <- 1 page CVPR format report
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── \_can_be_deleted   <- Trash bin (!! git ignored)
    │
    ├───noteboks           <- Notebooks used on Colab for training 
    │       ap_1_CNN_ResNet_Scattering.ipynb
    │       ap_2_feature_extraction_with_autoencoder.ipynb
    │
    ├───experiment_ae_features
    │       result_ae_features.csv
    │       result_without_ae_features.csv
    │
    ├───experiment_resnet
    │       result_resnet.csv
    │
    ├───experiment_scattering
    │       result_scattering.csv
    │
    └───src
            data.py        <- Defines data related stuffs, data loaders and transforms
            models.py      <- Defines models' architecture. 
            utils.py
            __init__.py




#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating your model
See the notebooks. The models used are defined in the src/models.py file. The tools to load and processed are defined in the src/data.py file.


#### Evaluating your model on the test set

Notebooks generate a file `result_*.csv` that I uploaded to the private kaggle competition website.

