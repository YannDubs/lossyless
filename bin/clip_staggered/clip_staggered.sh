#!/usr/bin/env bash

experiment="clip_staggered"
notes="
**Goal**: Add an entropy bottleneck to CLIP, and trains only entropy bottleneck.This pretrains the generic compressor that will be reused for all downstream datasets.
"

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
paths.pretrained.save=$pretrained_path
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

wait

# for featurizer
col_val_subset=""
python aggregate.py \
       experiment=$experiment  \
       patterns.predictor=null \
       $col_val_subset \
       agg_mode=[summarize_metrics]