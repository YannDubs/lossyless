#!/usr/bin/env bash

export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="STL10_understand_VIC"
notes="
**Goal**: Undersntad how to get VIC to 0.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# project and server kwargs
kwargs="
experiment=$experiment
timeout=$time
$add_kwargs
"

# experiment kwargs
kwargs="
$kwargs
is_only_feat=False
featurizer=neural_feat
architecture@encoder=resnet18
architecture@predictor=mlp_probe
data@data_feat=stl10_unlabeled
data@data_pred=stl10_aug
rate=H_hyper
trainer.max_epochs=100
data_feat.kwargs.batch_size=128
encoder.z_dim=64
featurizer.loss.beta_anneal=linear
rate.kwargs.warmup_k_epoch=1
optimizer@optimizer_feat=AdamW
optimizer_feat.kwargs.weight_decay=1e-6
optimizer_feat.kwargs.lr=1e-3
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=1e-6
optimizer_coder.kwargs.lr=3e-4
scheduler@scheduler_feat=expdecay100
scheduler@scheduler_coder=expdecay100
seed=1
data_pred.kwargs.batch_size=64
predictor.arch_kwargs.hid_dim=2048
predictor.arch_kwargs.dropout_p=0
predictor.arch_kwargs.n_hid_layers=2
optimizer@optimizer_pred=Adam
optimizer_pred.kwargs.weight_decay=1e-6
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=expdecay1000
evaluation.predictor.is_eval_train=True
"

# sweeping arguments
kwargs_hypopt_VIC="
distortion=VAE,VIC
featurizer.loss.beta=1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0
distortion.factor_beta=1
"

# sweeping arguments
kwargs_hypopt_BINCE="
distortion=BINCE
featurizer.loss.beta=1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0
distortion.factor_beta=1e-2
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_BINCE $kwargs_dep -m &

    sleep 60

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_VIC $kwargs_dep -m &

  done
fi

wait 
data="merged" # want to access both ther featurizer data and the  predictor data
python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]

