#!/usr/bin/env bash

script=`realpath $0`
base_dir=`dirname $script`/../..
export pretrained_path="$base_dir/hub"

is_use_pretrained='true' # whether to use pretrained model instead of running

if [[ "${is_use_pretrained}" == "true" ]]; then
    # the checkpoint is saved with git lfs, so very likely that you don't have the whole file in which case download manually
    min_size_ckpt=10000
    file_name="$base_dir/hub/best_featurizer.ckpt"
    size=$(wc -c <"$file_name")
    if [ $size -le $min_size_ckpt ]; then
        echo "Downloading the pretrained model ..."
        rm $file_name
        # this does not currently work because directory is private
        wget -P "$pretrained_path" https://github.com/YannDubs/lossyless/raw/main/hub/best_featurizer.ckpt
    fi
else
    echo "Running pretraining ..."
    # pretrain staggered model 
    `dirname $0`/clip_staggered.sh "$@"
fi


# datasets evaluated with accuracy
`dirname $0`/clip_staggered_stl10.sh "$@" &
sleep 60
`dirname $0`/clip_staggered_cars196.sh "$@" &
sleep 60
`dirname $0`/clip_staggered_imagenet.sh "$@" &
sleep 60
# `dirname $0`/clip_staggered_cifar10.sh "$@" &
# sleep 60
# `dirname $0`/clip_staggered_food101.sh "$@" &
# sleep 60
# `dirname $0`/clip_staggered_galaxy.sh "$@" &
# sleep 60
# `dirname $0`/clip_staggered_pcam.sh "$@" &
# sleep 60
# `dirname $0`/clip_staggered_stl10.sh "$@" &
# sleep 60

# # datasets evaluated with balanced accuracy
# `dirname $0`/clip_staggered_pets37.sh "$@" &
# sleep 60
# `dirname $0`/clip_staggered_caltech101.sh "$@" &
