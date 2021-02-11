#!/usr/bin/env bash

experiment="galaxy_featurizers"
notes=" "

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
is_only_feat=False
trainer.max_epochs=200
data@data_feat=galaxy64
architecture@encoder=resnet18
architecture@predictor=resnet18
evaluation.is_est_entropies=True
experiment=$experiment 
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
featurizer=ivae_hfac_b01,ivae_hfac_b1,ivae_hfac_b10,vae_hfac_b0.01,vae_hfac_b01,vae_hfac_b1,vae_hfac_b10,none,webp++,webp--
seed=1,2,3
"


kwargs_multi="
featurizer=ivae_hfac_b01vae_hfac_b0.01,none,webp++
seed=1
trainer.max_epochs=2
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi