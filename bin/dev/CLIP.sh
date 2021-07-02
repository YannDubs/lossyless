#!/usr/bin/env bash

experiment="jpeg"
notes="
**Goal**: compute jpeg size
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
featurizer=jpeg++
featurizer.quality=95
is_only_feat=True
data@data_pred=pcam
data@data_feat=pcam
trainer.max_epochs=30
data_feat.kwargs.dataset_kwargs.is_normalize=False
data_pred.kwargs.dataset_kwargs.is_normalize=False
$add_kwargs
"

kwargs_multi="" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
