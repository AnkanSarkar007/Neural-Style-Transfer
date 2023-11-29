# Neural Style Transfer with Mask-guided Autoencoder

## Overview

This project implements neural style transfer using a mask-guided autoencoder. The model is trained on a combination of content and style images, utilizing a generative adversarial network (GAN) with additional perceptual and style losses.

## Paper Reference

- [Neural Style Transfer with Mask-guided Autoencoder](https://arxiv.org/pdf/1805.09987.pdf)

## Dataset

- Style Dataset: [WikiArt Dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md)
- Content Dataset: [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)

## Table of Contents

1. [Dependencies](#dependencies)
2. [Training](#training)
3. [Testing](#testing)
4. [Presentation](#presentation)
5. [Description of python files](#description)

## Dependencies

Following are the majorly required python libraries for this model to work

1. Pytorch
2. Torchvision
3. Numpy
4. Matplotlib
5. utils
6. cv2

## Training

To train the model, run the following command:

```bash
python3 train_mask.py --dataset face --content-data data/content --style-data data/style --enc-model none --dec-model none --epochs 150 --lr-freq 60 --batch-size 56 --test-batch-size 24 --num-workers 8 --print-freq 200 --dropout 0.5 --g-optm adam --lr 0.002 --optm padam --d-lr 0.0002 --adam-b1 0.5 --weight-decay 0 --ae-mix mask --dise-model none --cla-w 1 --gan-w 1 --per-w 1 --gram-w 200 --cycle-w 0 --save-run debug_gan --gpuid 0 --train-dec --use-proj --dec-last tanh --trans-flag adin --ae-dep E5-E4 --base-mode c4 --st-layer 4w --seed 2017
```

One can further personalise the training as per the dataset by making use of other flags given in utils.py. The flags are self explanatory and also guven with a hepler argument

## Testing

```bash
python3 test_autoencoder.py  --content-data  data/test/content --style-data data/test/style --enc-model models/vgg_normalised_conv5_1.t7 --dec-model none  --dropout 0.5 --gpuid 0 --train-dec --dec-last tanh --trans-flag adin  --diag-flag batch --ae-mix mask --ae-dep E5-E4 --base-mode c4 --st-layer 4w --test-dp --save-image output --dise-model  none
```

## Presentation

- [Canva Presentation](https://www.canva.com/design/DAF0r_NG1NE/MkldKMbGKtw5gfjM4HkqIg/edit)

## Description of files

### [`folder.py`](behance/folder.py) and [`load_data.py`](behance/load_data.py)

These files examine the dataset folder, validate its directory structure, and apply transformations such as resizing and cropping. Random flipping is utilized for training data, and the final dataset is returned.

### [`utils.py`](behance/utils.py)

Centralizes all flags used in training and testing processes.

### [`net_utils.py`](behance/net_utils.py)

Defines the architecture of encoder, decoder, and discriminator layers. The specified configuration is crucial for model creation.

### [`make_loss.py`](behance/make_loss.py)

Encapsulates all loss functions used in the model.

### [`make_opt.py`](behance/make_opt.py)

Contains functions related to the Adam optimizer and its operations.

### [`my_autoencoder.py`](behance/my_autoencoder.py)

Defines the Autoencoder class, comprising both encoder and decoder components. Additionally, it contains the mask for the autoencoder and the AdaIn transformation.

### [`my_discriminator.py`](behance/my_discriminator.py)

Houses the Discriminator class, responsible for distinguishing between real and fake images. It is initialized with zero weights.

### [`train_mask.py`](behance/train_mask.py)

Integrates the autoencoder and discriminator for model training. The model is trained on batches of data, covering all possible combinations of classes in both content and style datasets. The trained model is then saved.

### [`test_autoencoder.py`](behance/test_autoencoder.py)

Loads the specified model and generates neural-style-transferred images for each combination of content and style images provided.

Code comments are thoughtfully placed throughout the files for enhanced understanding.

For sample output, see [`behance\output`](behance/output/)