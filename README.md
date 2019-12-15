# Single Image Deraining Using a Recurrent Dual-Attention-Residual Ensemble Network

# Abstract
   Single image deraining is an ill-posed inverse problem due to the presence of non-uniform rain shapes, directions, and densities in images. In this paper, we propose a Single Image Deraining Using a Recurrent Dual-Attention-Residual Ensemble Network(RDARNet). Extensive experiments demonstrate that the effect of removing rain and restoring texture details is greatly improved.

# Dataset
## Synthetic Datasets
   | Datasets | train | test |
   | :------- | -----: | ----: |
   | [Rain100L](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) | 200    | 100   |
   | [Rain100H](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) | 1800   | 100   |
   | [Rain800](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md)  | 700    | 100   |
   | [Rain12](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) |   | 12  | 
   

# Pre-trained Model
**We note that these models is trained on NVIDIA GeForce RTX2080Ti:**

| Datasets | Pre-trained model |
| :------ | :--------------- |
| Rain100H | [Rain100H model TAB](https://github.com/rainbowH/RDARENet/tree/master/codes/checkpoint/) |


# Requirements
   - python 3.6.8
   - opencv 4.1.2
   - pyotrch 1.0.0

# Usages
   - **Clone this repo**
   ```
      $ git clone https://github.com/rainbowH/RDARENet
      $ cd RDARENet
   ```

   - **Test**
   ```
      $ python RDARENet_test.py 
   ```
