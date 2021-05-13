#!/usr/bin/env bash

export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="dev_galaxy_balle"
notes="
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
rate=H_spatial
trainer.max_epochs=2
"

# VAE sweeping arguments
kwargs_hypopt_ivae="
featurizer=neural_rec
architecture@encoder=balle
distortion.kwargs.arch=balle
distortion.factor_beta=1.0
rate.factor_beta=1.0
featurizer.loss.beta=0
data_feat.kwargs.batch_size=128
encoder.z_dim=1024
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=3e-4
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=1e-5
optimizer_coder.kwargs.lr=1e-4
scheduler@scheduler_feat=expdecay100
scheduler@scheduler_coder=expdecay100
seed=0
optimizer@optimizer_pred=Adam
optimizer_pred.kwargs.weight_decay=1e-5
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=expdecay100
evaluation.featurizer.is_evaluate=False
trainer.precision=32
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs distortion=vae data@data_feat=galaxy64_unaug $kwargs_hypopt_ivae $kwargs_dep 

  done
fi
