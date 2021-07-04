#!/usr/bin/env bash


export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="STL10_dist_variation_featpred"
notes="
**Goal**: Different distortions on STL10: iNCE,VIC,VAE, predicted on features with MLP, ca. 100 runs for each config
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
+update_trainer_pred.max_epochs=100
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

# sweeping arguments
kwargs_hypopt_VIC="
distortion=VAE,VIC
hydra.sweeper.n_trials=200
hydra.sweeper.n_jobs=200
distortion.factor_beta=1.0
data_feat.kwargs.batch_size=tag(log,int(interval(32,128)))
encoder.z_dim=tag(log,int(interval(32,512)))
featurizer.loss.beta_anneal=linear,constant
featurizer.loss.beta=tag(log,interval(1e-8,1e0))
distortion.factor_beta=tag(log,interval(1e-5,100))
rate.kwargs.warmup_k_epoch=int(interval(0,3))
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,plateau
scheduler@scheduler_coder=cosine_restart,expdecay100,plateau_quick,unifmultistep100
seed=0,1,2,3,4
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
predictor.arch_kwargs.hid_dim=tag(log,int(interval(1024,4096)))
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
predictor.arch_kwargs.n_hid_layers=1,2
optimizer@optimizer_pred=SGD_likeadam,Adam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,expdecay1000,unifmultistep100
"

# sweeping arguments
kwargs_hypopt_BINCE="
distortion=BINCE
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=100
data_feat.kwargs.batch_size=tag(log,int(interval(64,512)))
encoder.z_dim=tag(log,int(interval(32,512)))
featurizer.loss.beta_anneal=linear,constant
featurizer.loss.beta=tag(log,interval(1e-8,1e-2))
distortion.factor_beta=tag(log,interval(1e-5,1e-1))
rate.kwargs.warmup_k_epoch=int(interval(0,3))
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,unifmultistep1000
scheduler@scheduler_coder=cosine_restart,expdecay100,unifmultistep1000,unifmultistep100
seed=0,1,2,3,4
distortion.kwargs.project_kwargs.out_shape=tag(log,interval(0.05,0.5))
distortion.kwargs.temperature=tag(log,interval(0.01,0.3))
distortion.kwargs.is_train_temperature=true,false
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
predictor.arch_kwargs.dropout_p=interval(0.,0.4)
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,unifmultistep100
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_BINCE $kwargs_dep -m &

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_VIC $kwargs_dep -m &

    sleep 7
  done
fi