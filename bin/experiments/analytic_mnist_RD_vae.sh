#!/usr/bin/env bash

experiment="analytic_mnist_RD_vae"
notes="
**Goal**: Understand how the gains changing at different points of the rate distortion curve for VAE and iVAE when looking at H[M(X)|Z] and the upperbound H[X|Z]
**Hypothesis**: should follow closely schematic polot. So gains compared to upper bound should be constant but actual gains can fall in between  
**Plot**: a RD curve for iVAE with invariance distortion, VAE with invariance distortion, and VAE with non invariance distoriton (i.e. upper bound), and write down theoretical gains 
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
data=analytic_mnist
evaluation.is_est_entropies=True
trainer.max_epochs=100
data.kwargs.dataset_kwargs.equivalence=[x_translation,y_translation]
$add_kwargs
"
#trainer.max_epochs=200

# every arguments that you are sweeping over
kwargs_multi="
distortion=ivae
loss.beta=0.001
seed=1
" 
#loss.beta=0.01,0.03,0.1,0.3,1,3,10,30,100

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi

#TODO plotting pipeline