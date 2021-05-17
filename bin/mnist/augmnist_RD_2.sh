#!/usr/bin/env bash

experiment="augmnist_RD_2"
notes="
**Goal**: RD curves for mnist that is augmented at test and training time (i.e. when we assume we know the augmentations),
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=mnist
experiment=$experiment 
timeout=$time
is_only_feat=False
featurizer=neural_rec
architecture@encoder=resnet18
architecture@predictor=resnet18
data@data_feat=mnist_aug
rate=H_hyper
trainer.max_epochs=100
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=vae
featurizer.loss.beta=0.0001,0.001,0.01,0.1,1,10,100
seed=1
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi


wait


