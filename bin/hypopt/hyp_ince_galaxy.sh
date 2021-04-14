#!/usr/bin/env bash

experiment="hyp_ince_galaxy"
notes="
**Goal**: Hyperparameter tuning for ince on the new galaxy dataset
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
data@data_feat=galaxy64
rate=H_hyper
trainer.max_epochs=50
+update_trainer_pred.max_epochs=100
distortion=ince
featurizer.loss.beta_anneal=linear
rate.kwargs.invertible_processing=diag
predictor.arch_kwargs.hid_dim=2048
$add_kwargs
"
#TODO increase epochs before pushing
#TODO chose best augmentation from `hyp_galaxy_augmentations`

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=50
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
data_feat.kwargs.batch_size=tag(log,int(interval(128,256)))
encoder.z_dim=tag(log,int(interval(128,1024)))
featurizer.loss.beta=tag(log,interval(3e-6,1e-3))
distortion.factor_beta=tag(log,interval(1e-5,1e-1))
optimizer@optimizer_feat=Adam,AdamW
rate.kwargs.warmup_k_epoch=int(interval(0,5))
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(5e-5,1e-3))
optimizer_coder.kwargs.weight_decay=tag(log,interval(3e-6,1e-4))
optimizer_coder.kwargs.lr=tag(log,interval(3e-4,1e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,unifmultistep1000
scheduler@scheduler_coder=cosine_restart,expdecay100,unifmultistep1000,unifmultistep100
seed=0,1,2,3,4,5,6,7,8,9
distortion.kwargs.project_kwargs.out_shape=tag(log,interval(0.05,0.5))
distortion.kwargs.temperature=tag(log,interval(0.01,0.2))
" 
# distortion.factor_beta : instead of deacreasing weight given to rate will increase weight given to distortion
# BATCH SIZE: for INCE it can be beneficial to use larger batches. THe issues is that this might be worst for other parts of the networks. SOme papers say using `is_lars=True` can mititgate the issue when using large batches
# SEED: here the seed is not optimized over because we are using random sampling (+ anyways it's ok to optimize over the initialization as long as it's done the same way for baselines also)
# the only different parameters with IVAE are the last 2 ones (all other hyperameters you can play around with them but I found that the current values were good)

# PREDICTOR
kwargs_multi="
$kwargs_multi
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
optimizer@optimizer_pred=SGD_likeadam,Adam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(5e-4,3e-3))
scheduler@scheduler_pred=cosine_restart,expdecay100,plateau_quick,unifmultistep1000
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
  for kwargs_dep in  ""        
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi

wait

col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_err','test/pred/loss','test/pred/err','train/pred/err']"
compare="dist"
data="merged" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       +summarize_RD_curves.data="${data}" \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="[]" \
       +plot_all_RD_curves.data="${data}" \
       +plot_all_RD_curves.rate_cols="${rate_cols}" \
       +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
       +plot_all_RD_curves.hue=$compare \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_optuna_hypopt] 

# !if you want an additional parameter from the configs use something like:
# +collect_data.kwargs.params_to_add.lr="optimizer_feat.kwargs.lr" \