#!/usr/bin/env bash

experiment="hyp_rates_mnist"
notes="
**Goal**: Understand how to best sweep over beta to generate RD curve for ivib. Find good beta, and good beta factors for rate and distortion
**Plot**: RD curve 
"


# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
data=mnist
encoder=cnn
seed=1
distortion=ivae
logger.wandb.project=hypopt
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
loss.beta=0.001,0.01,0.1,1,10,100
rate=MI_unitgaussian,MI_vamp,H_hyper,H_factorized
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3
    
  done
fi

#TODO plotting pipeline