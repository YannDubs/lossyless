#!/usr/bin/env bash


# Ensures that all data is downloaded
echo "Ensures that all data is downloaded."
`dirname $0`/../clip/download_data.sh "$@"

wait

# Pretrain our CLIP
echo "Pretrains the CLIP compressor."
`dirname $0`/clip_bottleneck_pretrain.sh "$@"

wait

# Evaluate our CLIP
echo "Evaluates the pretrained CLIP compressor with linear classifiers"
`dirname $0`/clip_bottleneck_linear_eval.sh "$@"
