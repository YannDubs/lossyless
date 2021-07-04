#!/usr/bin/env bash


# # Ensures that all data is downloaded
# echo "Ensures that all data is downloaded."
# bin/clip/download_data.sh

# wait

# ### OUR CLIP ###
echo "Pretrains the 3 different clips (for different values of beta)."
`dirname $0`/clip_bottleneck_pretrain.sh "$@"

# wait

# echo "Evaluates the pretrained CLIP models with linear classifiers"
# bin/clip/clip_bottleneck_linear_eval.sh "$@"

# wait 

# echo "Evaluates the pretrained CLIP models with linear classifiers"
# bin/clip/clip_raw_linear_eval.sh "$@"