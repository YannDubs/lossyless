#!/usr/bin/env bash

experiment=$prfx"hyp_epochs_selfsupervised"
notes=""

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
dataset=ImagenetDataset
model=CLIP_ViT
logger.kwargs.project=selfsupervised
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
trainer.max_epochs=3,10,30,50,100,150,200
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python bin/selfsupervised/selfsupervised.py $kwargs $kwargs_multi $kwargs_dep  -m &
    
  done
fi
