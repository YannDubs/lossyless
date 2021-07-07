#!/usr/bin/env bash

experiment="clip_bottleneck_linear_eval"
notes="
**Goal**: Linear evaluation of our CLIP compressors.
"


# parses special mode for running the script
source `dirname $0`/../utils.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
pretrained_path="$SCRIPTPATH"/../../pretrained/clip

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
featurizer=bottleneck_clip_lossyZ
$add_kwargs
"

kwargs_multi="
data@data_pred=stl10,cars196,caltech101,food101,pcam,pets37,cifar10,cifar100
" 


if [ "$is_plot_only" = false ] ; then
  for beta in   "5e-02" 
  do

    python utils/Z_linear_eval.py  $kwargs $kwargs_multi featurizer.loss.beta=$beta paths.pretrained.load=$pretrained_path/beta$beta  $kwargs_dep -m &

    sleep 30


  done
fi





