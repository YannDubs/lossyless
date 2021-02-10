#!/usr/bin/env bash

experiment="hyp_beta_mnist"
notes="
**Goal**: Understand how to best sweep over beta to generate RD curve. Find good beta, and good beta factors for rate and distortion
**Plot**: RD curve 
"

# Result
# should be sweeping 0.0001,0.001,0.01,0.03,0.1,0.3,1,3,10,30,100 and put distortion on log scale
# 0.01,0.1,1,10,100 probably ok for dev

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
rate=MI_unitgaussian
data=mnist
encoder=cnn
seed=1
logger.wandb.project=hypopt
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
loss.beta=0.0001,0.001,0.01,0.1,0.3,1,3,10,30,100,1000
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "distortion=ivae,ivib,ince rate=MI_unitgaussian" "distortion=ivae rate=MI_unitgaussian,MI_vamp,H_hyper,H_factorized"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3
    
  done
fi

#TODO plotting pipeline