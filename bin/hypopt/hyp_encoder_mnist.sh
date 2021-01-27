#!/usr/bin/env bash

experiment="hyp_encoder_mnist"
notes="
**Goal**: Hyperparameter tuning of the encoder on mnist
**Hypothesis**: cnn best in terms of performance and time
**Plot**: RD curve for the three encoder
"

# Result
# Resnet is  better but slower
# the performance gap between CNN and resnet is larger at large rate (small distortion)
# MLP is much worst
# CNN probably sufficient for hyp but resnet should be used for real results

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
rate=MI_unitgaussian
data=mnist
distortion=ivae
logger.wandb.project=hypopt
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
encoder=cnn,mlp,resnet
seed=1
loss.beta=0.01,0.1,1,3,10,30,100
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3
    
  done
fi

#TODO plotting pipeline