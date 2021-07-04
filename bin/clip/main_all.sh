#!/usr/bin/env bash


echo "Ensures that all data is downloaded"
`dirname $0`/download_data.sh "$@"

wait

### OUR CLIP ###
echo "Pretrains the 3 different clips (for different values of beta)"
`dirname $0`/clip_bottleneck_pretrain.sh "$@"

wait

echo "Evaluates the pretrained CLIP models with linear classifiers"
`dirname $0`/clip_bottleneck_linear_eval.sh "$@"

wait 

echo "Evaluates the pretrained CLIP models with MLP classifiers"
`dirname $0`/clip_bottleneck_mlp_eval.sh "$@"

wait 

### BASELINE CLIP ###
echo "Evaluates the pretrained CLIP models with linear classifiers"
`dirname $0`/clip_raw_linear_eval.sh "$@"

wait 

echo "Evaluates the pretrained CLIP models with MLP classifiers"
`dirname $0`/clip_raw_mlp_eval.sh "$@"