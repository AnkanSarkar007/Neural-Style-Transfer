# Neural-Style-Transfer

### Link - https://arxiv.org/pdf/1805.09987.pdf

dataset ->
https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md

To train ->
python3 train_mask.py  --dataset face --content-data  data/content --style-data  data/style --enc-model models/vgg_normalised_conv5_1.t7 --dec-model none  --epochs 150 --lr-freq 60 --batch-size 56 --test-batch-size 24 --num-workers 8  --print-freq 200 --dropout 0.5 --g-optm adam  --lr 0.002 --optm padam --d-lr 0.0002 --adam-b1 0.5  --weight-decay 0 --ae-mix mask --dise-model none --cla-w 1 --gan-w 1 --per-w 1 --gram-w 200 --cycle-w 0 --save-run debug_gan --gpuid 0 --train-dec --use-proj --dec-last tanh --trans-flag adin --ae-dep E5-E4 --base-mode c4  --st-layer 4w --seed 2017

To test ->
python3 test_autoencoder.py  --content-data  data/content --style-data data/style --enc-model models/vgg_normalised_conv5_1.t7 --dec-model none  --dropout 0.5 --gpuid 0 --train-dec --dec-last tanh --trans-flag adin  --diag-flag batch --ae-mix mask --ae-dep E5-E4 --base-mode c4 --st-layer 4w --test-dp --save-image output --dise-model  none

Presentation link ->
https://www.canva.com/design/DAF0r_NG1NE/MkldKMbGKtw5gfjM4HkqIg/edit
Paper2 implementation ->
https://arxiv.org/pdf/1705.04058.pdf

