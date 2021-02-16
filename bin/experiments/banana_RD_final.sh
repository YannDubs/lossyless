#!/usr/bin/env bash

experiment="banana_RD_final"
notes="
**Goal**: Run rate distortion curve for banana distribution when predicting representative
**Hypothesis**: Should be close to the estimated optimal rate distortion curve, and can approximate it using differential entropies.
"

# e.g. command: bin/experiments/banana_RD_final.sh -s vector  -t 720

# parses special mode for running the script
source `dirname $0`/../utils.sh


# most of these arguments are chose so as to replicate Fig.1.b. from "Non linear Transform coding" paper. 
# See their code here: https://github.com/tensorflow/compression/blob/master/models/toy_sources/toy_sources.ipynb
# only differences (as far as I can tell):
# - use batch norm
# - hidden dim for MLPs is 1024 instead of 100
# - train batch size is 8192 instead of 1024
# - not using soft rounding

# Encoder
encoder_kwargs="
architecture@encoder=fancymlp
encoder.z_dim=2
"

# Distortion
distortion_kwargs="
distortion.factor_beta=1
+architecture@distortion.kwargs=fancymlp
"

# Rate
rate_kwargs="
rate=H_factorized
rate.factor_beta=1
"

# Data
data_kwargs="
data@data_feat=bananaRot
data_feat.kwargs.batch_size=8192
data_feat.kwargs.val_size=100000
data_feat.kwargs.val_batch_size=16384
trainer.reload_dataloaders_every_epoch=True
"

#! check if work well without scheduler for the coder (you were previously using expdexay)
# Featurizer
general_kwargs="
is_only_feat=True
featurizer=neural_rec
optimizer@optimizer_feat=adam1e-3
scheduler@scheduler_feat=multistep
scheduler_feat.kwargs.MultiStepLR.milestones=[50,75,87]
optimizer@optimizer_coder=adam1e-3
scheduler@scheduler_coder=none
trainer.max_epochs=100
trainer.precision=32
evaluation.is_est_entropies=True
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
distortion=ivae,vae
featurizer.loss.beta=0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000
seed=1,2,3
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


# plot loaded model
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