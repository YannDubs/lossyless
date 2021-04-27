#!/usr/bin/env bash

experiment="dist" # should always be called with -m  dev
notes="
**Goal**: Checking that all models on distributions run without errors (in parallel)
**Hypothesis**: No errors
"

source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
data@data_feat=bananaRot
architecture@encoder=mlp
architecture@predictor=mlp
rate=H_factorized
featurizer=neural_feat
data_feat.kwargs.val_batch_size=1024
data_pred.kwargs.val_batch_size=1024
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=ivae,ince
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do
    
    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi
