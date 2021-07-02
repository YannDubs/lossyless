#!/usr/bin/env bash

experiment="clip_staggered_dev"
notes="
**Goal**: Add an entropy bottleneck to CLIP, and trains only entropy bottleneck.This pretrains the generic compressor that will be reused for all downstream datasets.
"

pretrained_path=`dirname $0`/../../hub

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=clip_staggered
experiment=$experiment 
timeout=$time
encoder.z_dim=512
is_only_feat=True
data@data_feat=coco
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=50
featurizer=bottleneck_clip_lossyZ
trainer.progress_bar_refresh_rate=30
paths.pretrained.save=$pretrained_path
$add_kwargs
"

kwargs_multi="" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep 

    sleep 3

  done
fi
