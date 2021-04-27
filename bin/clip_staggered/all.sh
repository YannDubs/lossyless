#!/usr/bin/env bash

# pretrain staggered model
`dirname $0`/clip_staggered.sh

# datasets evaluated with accuracy
`dirname $0`/clip_staggered_cars196.sh
`dirname $0`/clip_staggered_cifar10.sh
`dirname $0`/clip_staggered_food101.sh
`dirname $0`/clip_staggered_galaxy.sh
`dirname $0`/clip_staggered_imagenet.sh
`dirname $0`/clip_staggered_pcam.sh
`dirname $0`/clip_staggered_stl10.sh

# datasets evaluated with balanced accuracy
`dirname $0`/clip_staggered_pets37.sh
`dirname $0`/clip_staggered_caltech101.sh
