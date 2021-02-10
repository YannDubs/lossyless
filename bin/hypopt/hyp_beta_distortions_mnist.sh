#!/usr/bin/env bash

experiment="hyp_beta_distortions_mnist"
notes="
**Goal**: Understand how to best sweep over beta for different distortions to generate RD curve. 
**Plot**: RD curve 
"

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
logger.wandb.project=hypopt
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
loss.beta=0.001,0.01,0.1,0.3,1,3,10,30,100
" 

distortion=ivae,ivib,ince
if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "+parallel=distortions" "+parallel=idistortions"
  do

    # force use of parallel.py because small
    python parallel.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3
    
  done
fi

#TODO plotting pipeline