#!/usr/bin/env bash

experiment="dist" # should always be called with -m dev
notes="
**Goal**: Checking that all models on distributions run without errors (in parallel)
**Hypothesis**: No errors
"

source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
+mode=dev
timeout=$time
data=bananaRot
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
encoder=mlp
distortion=ivib,ivae,ince
rate=H_factorized
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do
    
    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m 
    
  done
fi

#TODO plotting pipeline