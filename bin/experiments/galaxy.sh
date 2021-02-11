#!/usr/bin/env bash

experiment="galaxy_RD"
notes=" "

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
data=galaxy64
experiment=$experiment
timeout=$time
encoder=resnet
rate=MI_unitgaussian
evaluation.is_est_entropies=True
trainer.max_epochs=200
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=vae
loss.beta=0.01,0.1,1.
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