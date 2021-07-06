#!/usr/bin/env bash


echo "Ensures that all data is downloaded"
`dirname $0`/download_data.sh "$@"

wait

### OUR CLIP ###
echo "Pretrains our CLIP compressor."
`dirname $0`/clip_bottleneck_pretrain.sh "$@"

wait

echo "Evaluates our pretrained CLIP compressor with linear classifiers"
`dirname $0`/clip_bottleneck_linear_eval.sh "$@"

wait 

echo "Evaluates our pretrained CLIP compressor with MLP classifiers"
`dirname $0`/clip_bottleneck_mlp_eval.sh "$@"

wait 

### BASELINE CLIP ###
echo "Evaluates raw pretrained CLIP model with linear classifiers"
`dirname $0`/clip_raw_linear_eval.sh "$@"

wait 

echo "Evaluates raw pretrained CLIP model with MLP classifiers"
`dirname $0`/clip_raw_mlp_eval.sh "$@"