#!/usr/bin/env bash

export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="dev_galaxy256"
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
trainer.max_epochs=100
"

# VAE sweeping arguments
kwargs_hypopt_ivae="
featurizer=neural_rec
distortion.factor_beta=1.0
rate.factor_beta=1.0
featurizer.loss.beta=0
data_feat.kwargs.batch_size=64
encoder.z_dim=16384
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=3e-4
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=1e-5
optimizer_coder.kwargs.lr=1e-4
scheduler@scheduler_feat=expdecay100
scheduler@scheduler_coder=expdecay100
seed=0
+data_pred.kwargs.batch_size=64
optimizer@optimizer_pred=Adam
optimizer_pred.kwargs.weight_decay=1e-5
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=expdecay100
evaluation.featurizer.is_evaluate=False
architecture@encoder=balle 
distortion.kwargs.arch=balle 
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs distortion=ivae data@data_feat=galaxy256 $kwargs_hypopt_ivae $kwargs_dep -m &

  done
fi
