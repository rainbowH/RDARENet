# Single Image Deraining Using a Recurrent Dual-Attention-Residual Ensemble Network

# Abstract
   Single image deraining is an ill-posed inverse problem due to the presence of non-uniform rain shapes, directions, and densities in images. In this paper, we propose a Single Image Deraining Using a Recurrent Dual-Attention-Residual Ensemble Network(RDARNet). Extensive experiments demonstrate that the effect of removing rain and restoring texture details is greatly improved.

<p align='center'><img src='./materials/RDARNet.png' height="600px"/></p>

# Dataset
## Synthetic Datasets
   | Datasets | train | test |
   | :------- | -----: | ----: |
   | [Rain100L](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) | 200    | 100   |
   | [Rain100H](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) | 1800   | 100   |
   | [Rain800](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md)  | 700    | 100   |
   | [Rain12](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) |   | 12  | 
   
   
## Real-World Datasets
   | Datasets | #test |
   | :------- | :-----: |
   | [Real-World](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) |67 |

# Pre-trained Model
**We note that these models is trained on NVIDIA GeForce RTX2080Ti:**

| Datasets | Pre-trained model |
| :------ | :--------------- |
| Rain100L | [Rain100L model](https://github.com/rainbowH/RDARENet/tree/master/codes/checkpoint/) |
| Rain100H | [Rain100H model TAB](https://github.com/rainbowH/RDARENet/tree/master/codes/checkpoint/) |
| Rain800 | [Rain800 model TAB](https://github.com/rainbowH/RDARENet/tree/master/codes/checkpoint/) |

# Requirements
   - python 3.6.5
   - opencv 3.4.2
   - pyotrch 

# Usages
   - **Clone this repo**
   ```
      $ git clone https://github.com/rainbowH/RDARENet
      $ cd RDARENet
   ```

   - **Test**
   ```
      $ python RDARENet_test.py --is_testing True
		      --train_dataset Rain100L
		      --test_dataset Rain100L
		      --trainset_size 200
		      --testset_size 100
		      --batch_size 32
   ```
