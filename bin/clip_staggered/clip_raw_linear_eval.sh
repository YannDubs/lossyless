#!/usr/bin/env bash

experiment="clip_raw_linear_eval"
notes="
**Goal**: Linear evaluation of our CLIP compressor at different points on RI curve.
"


# parses special mode for running the script
source `dirname $0`/../utils.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
pretrained_path="$SCRIPTPATH"/../../hub

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger=none
experiment=$experiment 
timeout=$time
encoder.z_dim=512
data@data_feat=coco
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=30
data_feat.kwargs.num_workers=4
featurizer=clip_freeze
featurizer.is_use_init=True
$add_kwargs
"

kwargs_multi="
data@data_pred=stl10,cars196,stl10,caltech101,food101,pcam,pets37,cifar10,cifar100,imagenet
" 




if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""   
  do

    python utils/Z_linear_eval.py  $kwargs $kwargs_multi  $kwargs_dep -m &

  done
fi





