#!/usr/bin/env bash

experiment="hyp_ince_stl10"
notes="
**Goal**: Hyperparameter tuning for ince on stl10
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
is_only_feat=False
featurizer=neural_feat
architecture@encoder=resnet18
architecture@predictor=mlp_probe
data@data_feat=stl10unlabeled
data@data_pred=stl10_aug
rate=H_hyper
trainer.max_epochs=100
distortion=ince
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hypopt=multi_optuna
hydra.sweeper.optuna_config.n_trials=225
hydra.sweeper.optuna_config.n_jobs=75
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/feat/rate]
hydra.sweeper.optuna_config.sampler=random
"

# here the random sampler means that we are not actually doing smart hyperparametr tuning use `nsgaii` if you want
# n_trials is total trials and `n_jobs` are the ones to run in batch (when random it only changes compute but it's important when using other samplers)
# monitor is what you want to optimize ( only useful for plotting when using random sampler, but improtant otherwise)

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=tag(log,int(interval(32,1024)))
encoder.z_dim=tag(log,int(interval(16,512)))
featurizer.loss.beta=tag(log,interval(1e-8,1e2))
featurizer.loss.beta_anneal=linear,constant
rate.kwargs.warmup_k_epoch=int(interval(0,5))
rate.kwargs.invertible_processing=null,diag
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer_feat.kwargs.is_lars=true,false
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,3e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,plateau,unifmultistep1000
scheduler@scheduler_coder=cosine_restart,expdecay100,plateau_quick,unifmultistep1000,unifmultistep100
seed=0,1,2,3,4
distortion.kwargs.project_kwargs.out_shape=tag(log,interval(0.01,0.5))
distortion.kwargs.temperature=tag(log,interval(0.01,0.3))
" 
# BATCH SIZE: for INCE it can be beneficial to use larger batches. THe issues is that this might be worst for other parts of the networks. SOme papers say using `is_lars=True` can mititgate the issue when using large batches
# SEED: here the seed is not optimized over because we are using random sampling (+ anyways it's ok to optimize over the initialization as long as it's done the same way for baselines also)
# the only different parameters with IVAE are the last 2 ones (all other hyperameters you can play around with them but I found that the current values were good)

# PREDICTOR
kwargs_multi="
$kwargs_multi
data_pred.kwargs.batch_size=tag(log,int(interval(16,128)))
predictor.arch_kwargs.hid_dim=tag(log,int(interval(512,4096)))
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
predictor.arch_kwargs.n_hid_layers=1,2
optimizer@optimizer_pred=SGD_likeadam,Adam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,expdecay1000,unifmultistep100
" 

# comment

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""        
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi
