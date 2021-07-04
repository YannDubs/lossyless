#!/usr/bin/env bash

experiment=$prfx"test"
notes="
**Goal**: Checking that everything is working.
"

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
mode=dev
trainer.max_epochs=1
architecture@encoder=mlp_probe
rate=H_hyper
data@data_feat=mnist_aug
data@data_pred=mnist_aug
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=VIC,BINCE
" 

if [ "$is_plot_only" = false ] ; then
    # add kwargs if parameters have dependecies, so you cannot use Hydra's multirun (they will run in parallel in the background). 
    # e.g. `for kwargs1 in "a=1,2 b=3,4" "a=11,12 b=13,14" `
  for kwargs_dep in  "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m 

    wait
    
  done
fi
