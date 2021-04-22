#!/usr/bin/env bash

experiment="banana_viz_nce"
notes="
**Goal**: Run INCE on banana distributions to get nice figures.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# Encoder
encoder_kwargs="
architecture@encoder=fancymlp
encoder.z_dim=2
"

# Distortion
distortion_kwargs="
distortion=ince
distortion.factor_beta=1
distortion.kwargs.effective_batch_size=null
"

# Rate
rate_kwargs="
rate=H_factorized
rate.factor_beta=1
"

# Data
data_kwargs="
data_feat.kwargs.batch_size=1024
data_feat.kwargs.val_size=100000
data_feat.kwargs.val_batch_size=2048
trainer.reload_dataloaders_every_epoch=True
"

# Featurizer
general_kwargs="
is_only_feat=True
featurizer=neural_rec
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.lr=1e-3
scheduler@scheduler_feat=unifmultistep1000
optimizer@optimizer_coder=Adam
scheduler@scheduler_coder=none
optimizer_coder.kwargs.lr=1e-3
trainer.max_epochs=100
trainer.precision=32
"

kwargs="
logger.kwargs.project=banana
experiment=$experiment 
$encoder_kwargs
$distortion_kwargs
$rate_kwargs
$data_kwargs
$general_kwargs
timeout=${time}
$add_kwargs
"

kwargs_multi="
data@data_feat=banana_rot
distortion=ince
featurizer.loss.beta=0.7
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
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

col_val_subset=""
python load_pretrained.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       server=local \
       trainer.gpus=0 \
       $kwargs_multi \
       load_pretrained.mode=[maxinv_distribution_plot,codebook_plot] \
       -m