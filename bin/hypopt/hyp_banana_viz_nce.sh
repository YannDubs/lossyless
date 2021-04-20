#!/usr/bin/env bash

experiment="hyp_banana_viz_nce"
notes="
**Goal**: Run banana models for plotting, when using NCE.
"

# e.g. command: bin/experiments/banana_viz_nce.sh -s vector -t 360

# parses special mode for running the script
source `dirname $0`/../utils.sh


# most of these arguments are chose so as to replicate Fig.1.b. from "Non linear Transform coding" paper. 
# See their code here: https://github.com/tensorflow/compression/blob/master/models/toy_sources/toy_sources.ipynb
# only differences (as far as I can tell):
# - use batch norm
# - hidden dim for MLPs is 1024 instead of 100
# - train batch size is 8192 instead of 1024
# - not using soft rounding
# - 200 epochs and decrease lr at 120

# Encoder
encoder_kwargs="
architecture@encoder=fancymlp
encoder.z_dim=2
"
#TODO use mlp prove instead of fancy mlpo as it probably won't make a difference

# Distortion
distortion_kwargs="
distortion=ince
distortion.factor_beta=1
distortion.kwargs.effective_batch_size=null
"
#TODO you can remoce effective_batch_size = null

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

#! check if work well without scheduler for the coder (you were previously using expdexay)
# Featurizer
general_kwargs="
is_only_feat=True
featurizer=neural_rec
featurizer.loss.beta_anneal=constant
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.lr=1e-3
scheduler@scheduler_feat=multistep
scheduler_feat.kwargs.MultiStepLR.milestones=[50,75,87,120]
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.lr=1e-3
scheduler@scheduler_coder=none
trainer.max_epochs=200
trainer.precision=32
logger.kwargs.project=hypopt
"
#TODO scheculer feat can just use unifmultistep10000

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
distortion=ince
featurizer.loss.beta=0.01,0.1,1,3,10
" 



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "distortion.kwargs.is_train_temperature=False distortion.kwargs.temperature=0.1" "" "featurizer.loss.beta_anneal=constant"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3

  done
fi

wait 

# for featurizer
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss']"
compare="dist"
python aggregate.py \
       experiment=$experiment  \
       collect_data.predictor=null \
       $col_val_subset \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="${distortion_cols}" \
       +plot_all_RD_curves.rate_cols="${rate_cols}" \
       +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
       +plot_all_RD_curves.logbase_x=2 \
       +plot_all_RD_curves.hue=$compare \
       +plot_invariance_RD_curve.noninvariant='vae' \
       +plot_invariance_RD_curve.logbase_x=2 \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_invariance_RD_curve]

col_val_subset=""
python load_pretrained.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       server=local \
       trainer.gpus=0 \
       +load_pretrained.codebook_plot.is_plot_codebook=False \
       $kwargs_multi \
       load_pretrained.mode=[codebook_plot] \
       -m
