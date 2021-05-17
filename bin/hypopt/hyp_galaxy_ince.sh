#!/usr/bin/env bash

export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="hyp_galaxy_ince"
notes="
"
# parses special mode for running the script
source `dirname $0`/../utils.sh

# project and server kwargs
kwargs="
experiment=$experiment
timeout=$time
is_only_feat=False
rate=H_hyper
trainer.max_epochs=100
data_feat.kwargs.batch_size=64
data@data_feat=galaxy64
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=20
hydra.sweeper.n_jobs=20
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

kwargs_multi="
$kwargs_hypopt
encoder.z_dim=512
distortion=ince
featurizer=neural_feat
distortion.kwargs.is_train_temperature=true
distortion.kwargs.project_kwargs.out_shape=0.2,0.5,0.99
architecture@encoder=resnet18
distortion.factor_beta=1.0
rate.factor_beta=1.0
featurizer.loss.beta=0
seed=0
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=1e-4
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=1e-5
optimizer_coder.kwargs.lr=3e-4
scheduler@scheduler_feat=expdecay100
scheduler@scheduler_coder=expdecay100
architecture@predictor=mlp_probe
optimizer@optimizer_pred=Adam
optimizer_pred.kwargs.weight_decay=1e-5
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=expdecay100
predictor.arch_kwargs.dropout_p=0.3
featurizer.is_on_the_fly=false
evaluation.featurizer.is_evaluate=False
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

  done
fi
