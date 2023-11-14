# Neural-Style-Transfer

### Link - https://arxiv.org/pdf/1805.09987.pdf

dataset ->
https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md

To test ->
python3 test_autoencoder.py  --content-data  data/content --style-data data/style --enc-model models/vgg_normalised_conv5_1.t7 --dec-model none  --dropout 0.5 --gpuid 0 --train-dec --dec-last tanh --trans-flag adin  --diag-flag batch --ae-mix mask --ae-dep E5-E4 --base-mode c4 --st-layer 4w --test-dp --save-image output/face_mask --dise-model  none
