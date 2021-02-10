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
logger.name=false
callbacks.additional=[]
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
encoder=mlp
distortion=ivae
rate=H_hyper
data=mnist
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep 
    
  done
fi

#TODO plotting pipeline