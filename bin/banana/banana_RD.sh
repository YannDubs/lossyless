#!/usr/bin/env bash

experiment="banana_RD"
notes="
**Goal**: Run rate distortion curve for rotated banana distribution.
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
trainer.max_epochs=100
trainer.precision=32
architecture@predictor=mlp_probe
optimizer@optimizer_pred=Adam
scheduler@scheduler_pred=unifmultistep100
optimizer_pred.kwargs.lr=3e-4
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
featurizer.loss.beta=0.0001,0.001,0.01,0.03,0.1,0.3,1,3,10
seed=1,2,3
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "distortion=VAE encoder.z_dim=2"  "distortion=VIC encoder.z_dim=1" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi

wait


# for featurizer
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/pred/loss']"
compare="dist"
data="merged" # want to access both ther featurizer data and the  predictor data
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       +summarize_RD_curves.data="${data}" \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="${distortion_cols}" \
       +plot_all_RD_curves.data="${data}" \
       +plot_all_RD_curves.rate_cols="${rate_cols}" \
       +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
       +plot_all_RD_curves.logbase_x=2 \
       +plot_all_RD_curves.hue=$compare \
       +plot_invariance_RD_curve.data="${data}" \
       +plot_invariance_RD_curve.noninvariant='VAE' \
       +plot_invariance_RD_curve.logbase_x=2 \
       +plot_invariance_RD_curve.desirable_distortion="test/pred/loss" \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_invariance_RD_curve]

# plot loaded model
col_val_subset=""
python utils/load_pretrained.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       server=none \
       trainer.gpus=0 \
       $kwargs_multi \
       load_pretrained.mode=[maxinv_distribution_plot,codebook_plot] \
       -m