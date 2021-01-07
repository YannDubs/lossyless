#!/usr/bin/env bash

name="test_imgs"
notes="
**Goal**: Debugging all models on images at once (in parallel)
**Hypothesis**: No errors
"

source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
name=$name 
+mode=debug
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
encoder=cnn,mlp,resnet
distortion=ivib,ivae,ince,taskvib,vae,vib,nce
rate=H_factorized,H_hyper,MI_unitgaussian,MI_vamp
data=miniMnist
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python main.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m 
    
  done
fi

#TODO plotting pipeline