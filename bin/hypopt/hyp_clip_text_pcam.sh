#!/usr/bin/env bash

experiment="hyp_clip_text_pcam"
notes="
**Goal**: Test and tune hyperaparmeters for finetuning clip with text
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
encoder.z_dim=512
is_only_feat=False
architecture@predictor=mlp_probe
data@data_feat=coco
data@data_pred=pcam
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=50
+update_trainer_pred.max_epochs=150
featurizer=bottleneck_clip
data_feat.kwargs.num_workers=4
featurizer.is_on_the_fly=false
optimizer@optimizer_pred=Adam
featurizer.loss.beta_anneal=linear
rate.kwargs.invertible_processing=diag
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=30
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

# here the random sampler means that we are not actually doing smart hyperparametr tuning use `nsgaii` if you want
# n_trials is total trials and `n_jobs` are the ones to run in batch (when random it only changes compute but it's important when using other samplers)
# monitor is what you want to optimize ( only useful for plotting when using random sampler, but improtant otherwise)

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
featurizer.loss.beta=tag(log,interval(3e-6,1e-3))
distortion.factor_beta=1e-3
seed=0,1,2,3,4,5,6,7,8,9
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.weight_decay=5e-5
optimizer_feat.kwargs.lr=1e-4
optimizer_coder.kwargs.weight_decay=5e-4
optimizer_coder.kwargs.lr=5e-4
scheduler@scheduler_feat=expdecay100
scheduler@scheduler_coder=expdecay100
" 
# distortion.factor_beta : instead of deacreasing weight given to rate will increase weight given to distortion
# BATCH SIZE: for INCE it can be beneficial to use larger batches. THe issues is that this might be worst for other parts of the networks. SOme papers say using `is_lars=True` can mititgate the issue when using large batches
# SEED: here the seed is not optimized over because we are using random sampling (+ anyways it's ok to optimize over the initialization as long as it's done the same way for baselines also)
# the only different parameters with IVAE are the last 2 ones (all other hyperameters you can play around with them but I found that the current values were good)

# PREDICTOR
kwargs_multi="
$kwargs_multi
predictor.arch_kwargs.dropout_p=interval(0.,0.4)
optimizer@optimizer_pred=Adam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(7e-4,3e-3))
scheduler@scheduler_pred=cosine_restart,plateau_quick,unifmultistep1000
" 


# kwargs_multi="
# trainer.max_epochs=1
# ++update_trainer_pred.max_epochs=1
# mode=dev
# +update_trainer_pred.limit_val_batches=1.0
# +update_trainer_pred.limit_test_batches=0.5
# evaluation.featurizer.is_evaluate=True
# data_feat.kwargs.batch_size=64
# data_pred.kwargs.batch_size=64
# optimizer_pred.kwargs.lr=1e-3
# "

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi
