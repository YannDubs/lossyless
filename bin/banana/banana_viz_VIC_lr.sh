#!/usr/bin/env bash

experiment="banana_viz_VIC_lr"
notes="
**Goal**: Run VAE and VIC on banana distributions with rotation invariance to get nice figures.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh


# Encoder
encoder_kwargs="
architecture@encoder=mlp_fancy
"

# Distortion
distortion_kwargs="
distortion.factor_beta=1
architecture@distortion.kwargs=mlp_fancy
"
# like in their paper we are using softplus activation which gives slightly more smooth decision boundaries 

# Rate
rate_kwargs="
rate=H_factorized
rate.factor_beta=1
"

# Data
data_kwargs="
data@data_feat=banana_rot
trainer.reload_dataloaders_every_epoch=True
"

# Featurizer
general_kwargs="
is_only_feat=False
featurizer=neural_feat
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.lr=3e-4
scheduler@scheduler_feat=expdecay1000
optimizer@optimizer_coder=Adam
scheduler@scheduler_coder=expdecay100
optimizer_coder.kwargs.lr=3e-4
trainer.max_epochs=20
trainer.precision=32
architecture@predictor=mlp_probe
optimizer@optimizer_pred=Adam
scheduler@scheduler_pred=unifmultistep100
optimizer_pred.kwargs.lr=3e-4
featurizer.loss.beta_anneal=constant
distortion=VIC 
encoder.z_dim=1
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
featurizer.loss.beta=0.07
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "" "optimizer_feat.kwargs.lr=1e-3,1e-4,3e-4" "featurizer.loss.beta_anneal=linear" "scheduler@scheduler_feat=unifmultistep1000" "featurizer.loss.beta=0.1,0.05" "encoder.z_dim=2" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3

  done
fi
