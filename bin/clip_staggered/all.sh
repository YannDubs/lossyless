#!/usr/bin/env bash

export pretrained_path=`dirname $0`/../../hub

# pretrain staggered model (only uncomment to rerun)
#`dirname $0`/clip_staggered.sh "$@"

# datasets evaluated with accuracy
`dirname $0`/clip_staggered_stl10.sh "$@"
`dirname $0`/clip_staggered_cars196.sh "$@"
`dirname $0`/clip_staggered_cifar10.sh "$@"
`dirname $0`/clip_staggered_food101.sh "$@"
`dirname $0`/clip_staggered_galaxy.sh "$@"
`dirname $0`/clip_staggered_imagenet.sh "$@"
`dirname $0`/clip_staggered_pcam.sh "$@"
`dirname $0`/clip_staggered_stl10.sh "$@"

# datasets evaluated with balanced accuracy
`dirname $0`/clip_staggered_pets37.sh "$@"
`dirname $0`/clip_staggered_caltech101.sh "$@"
