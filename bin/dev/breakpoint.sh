#!/usr/bin/env bash

experiment=$prfx"breakpoint"
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
rate=H_hyper
data@data_feat=cifar10
featurizer=neural_feat
trainer.max_epochs=100
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

