#!/usr/bin/env bash

experiment="dev_dist"
notes="
**Goal**: Checking that all models on distributions run without errors (in parallel)
**Hypothesis**: No errors
"

source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
+mode=dev
timeout=60
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
encoder=mlp
distortion=ivib,ivae,ince,vae
rate=H_factorized,H_hyper,MI_unitgaussian,MI_vamp
data=bananaRot,bananaXtrnslt,bananaYtrnslt
" 

kwargs_multi="
encoder=mlp
distortion=ivib
rate=H_factorized
data=bananaRot
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python main.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m 
    
  done
fi

#TODO plotting pipeline