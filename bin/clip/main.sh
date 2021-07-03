#!/usr/bin/env bash


# Ensures that all data is downloaded
bin/clip/download_data.sh

wait

### OUR CLIP ###
# Pretrains the 3 different clips (for different values of beta)
bin/clip/clip_bottleneck_pretrain.sh

wait

# Evaluates the pretrained CLIP models with linear classifiers
bin/clip/clip_bottleneck_linear_eval.sh

wait 

# Evaluates the pretrained CLIP models with MLP classifiers
bin/clip/clip_bottleneck_mlp_eval.sh

wait 

### BASELINE CLIP ###
# Evaluates the pretrained CLIP models with linear classifiers
bin/clip/clip_raw_linear_eval.sh

wait 

# Evaluates the pretrained CLIP models with MLP classifiers
bin/clip/clip_raw_mlp_eval.sh