#!/usr/bin/env bash

experiment="banana_viz_BINCE"
notes="
**Goal**: Run BINCE on banana distributions with rotation invariance to get nice figures.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# Encoder
encoder_kwargs="
architecture@encoder=mlp_fancy
encoder.z_dim=1
"

# Distortion
distortion_kwargs="
distortion=BINCE
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
trainer.reload_dataloaders_every_epoch=True
"

# Featurizer
general_kwargs="
is_only_feat=False
featurizer=neural_feat
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.lr=1e-3
scheduler@scheduler_feat=expdecay1000
optimizer@optimizer_coder=Adam
scheduler@scheduler_coder=expdecay100
optimizer_coder.kwargs.lr=3e-4
trainer.max_epochs=100
trainer.precision=32
architecture@predictor=mlp_probe
optimizer@optimizer_pred=Adam
scheduler@scheduler_pred=unifmultistep100
optimizer_pred.kwargs.lr=1e-3
featurizer.loss.beta_anneal=constant
"

kwargs="
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
distortion=BINCE
featurizer.loss.beta=0.6
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
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.predictor=null \
       $col_val_subset \
       agg_mode=[summarize_metrics]

col_val_subset=""
python utils/load_pretrained.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       server=local \
       trainer.gpus=0 \
       $kwargs_multi \
       +load_pretrained.codebook_plot.is_plot_codebook=false \
       load_pretrained.mode=[codebook_plot] \
       -m