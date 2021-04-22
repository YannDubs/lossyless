#!/usr/bin/env bash

experiment="clip_staggered"
notes="
**Goal**: Add an entropy bottleneck to CLIP, and trains only entropy bottleneck.
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
$add_kwargs
"

# FEATURIZER
# all hyperparameters (it's quite robust as you are only training a few hyperparameters)
# the only really important ones are beta and factor_beta
kwargs_multi="
data_feat.kwargs.batch_size=32
featurizer.loss.beta=5e-2
distortion.factor_beta=1e-3
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=3e-4
scheduler@scheduler_feat=plateau_quick
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=3e-6
optimizer_coder.kwargs.lr=1e-4
scheduler@scheduler_coder=expdecay100
" 

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