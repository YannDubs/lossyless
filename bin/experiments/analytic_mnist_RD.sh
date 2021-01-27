#!/usr/bin/env bash

experiment="analytic_mnist_RD"
notes="
**Goal**: Understand how the gains chanin at different points of the rate distortion curve
**Hypothesis**: in theory the gains at different RD points should be constant
**Plot**: a RD curve for each non invariant loss, with dotted line for theoretical (i.e. constant reduction), and the invariant RD curve
"

#! UPDATE: this is not analytic actually, all we can say is the maximal gains that you can have not the actual gains that you can have.

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
encoder=resnet
rate=H_factorized
data=mnist
evaluation.is_est_entropies=True
trainer.max_epochs=200
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=ivae,vae,taskvib
loss.beta=0.01,0.03,0.1,0.3,1,3,10,30,100
seed=1,2,3
" 
# ivib,ivae,ince,vae,nce,vib,taskvib


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi

#TODO plotting pipeline