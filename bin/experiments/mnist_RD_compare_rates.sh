#!/usr/bin/env bash

experiment="mnist_RD_compare_rates"
notes="
**Goal**: comparing the variational bounds of rates
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
encoder=cnn
distortion=ivae
data=analytic_mnist
evaluation.is_est_entropies=True
trainer.max_epochs=100
$add_kwargs
"
# encoder=resnet
# trainer.max_epochs=200

# every arguments that you are sweeping over
kwargs_multi="
rate=H_factorized,H_hyper,MI_vamp,MI_unitgaussian
loss.beta=0.01,0.03,0.1,0.3,1,10,100
seed=1
" 
# loss.beta=0.01,0.03,0.1,0.3,1,3,10,30,100
# seed=1,2,3

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi

#TODO plotting pipeline