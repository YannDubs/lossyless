#!/usr/bin/env bash

experiment=$prfx"breakpoint_dist"
notes="
**Goal**: Script that can be used when putting breakpoints.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
+mode=dev
logger=none
callbacks.additional=[]
architecture@encoder=mlp
architecture@predictor=mlp
distortion=ivae
rate=H_factorized
data@data_feat=banana
+data@data_pred=bananaRot
data_feat.kwargs.val_batch_size=1024
data_pred.kwargs.val_batch_size=1024
featurizer=neural_rec
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep 
    
  done
fi
