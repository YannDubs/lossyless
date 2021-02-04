#!/usr/bin/env bash

experiment="imgs" # should always be called with -m dev
notes="
**Goal**: Checking that all models on images run without errors (in parallel)
**Hypothesis**: No errors
"

source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
+mode=dev
logger.name=false
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
encoder=cnn,mlp,resnet
distortion=ivib,ivae,ince,taskvib,vae,vib,nce
rate=H_factorized,H_hyper,MI_unitgaussian,MI_vamp
data=mnist
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi

#TODO plotting pipeline