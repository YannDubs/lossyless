#!/usr/bin/env bash

experiment="hyp_ivib_mnist"
notes="
**Goal**: Understand how to best sweep over beta to generate RD curve for ivib. Find good beta, and good beta factors for rate and distortion
**Plot**: RD curve 
"

# RESULT
# not working!!!! rate=0


# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
rate=H_factorized
data=mnist
encoder=cnn
seed=1
distortion=ivib
logger.wandb.project=hypopt
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
loss.beta=0.01,0.03,0.1,0.3,1,3,10,30,100
+parallel=lrs
" 
kwargs_multi="
loss.beta=0.01,0.03,0.1,0.3,1,3,10,30,100
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3
    
  done
fi

#TODO plotting pipeline