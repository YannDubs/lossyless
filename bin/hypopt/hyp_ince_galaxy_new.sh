#!/usr/bin/env bash

experiment="hyp_ince_galaxy_new"
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
data@data_feat=galaxy64
rate=H_hyper
trainer.max_epochs=50
+update_trainer_pred.max_epochs=100
$add_kwargs
"
#TODO increase epochs before pushing

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=10
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/loss,test/comm/rate]
"

# here the random sampler means that we are not actually doing smart hyperparametr tuning use `nsgaii` if you want
# n_trials is total trials and `n_jobs` are the ones to run in batch (when random it only changes compute but it's important when using other samplers)
# monitor is what you want to optimize ( only useful for plotting when using random sampler, but improtant otherwise)

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=128
encoder.z_dim=128
featurizer.loss.beta=tag(log,interval(1e-3,3e-2))
distortion.factor_beta=tag(log,interval(1e-3,3e-2))
seed=0,1,2,3,4,5,6,7,8,9
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=3e-4
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=1e-6
optimizer_coder.kwargs.lr=1e-4
scheduler@scheduler_feat=expdecay1000,expdecay100,unifmultistep100,unifmultistep1000
scheduler@scheduler_coder=expdecay100
distortion.kwargs.is_train_temperature=false
" 
# distortion.factor_beta : instead of deacreasing weight given to rate will increase weight given to distortion
# BATCH SIZE: for INCE it can be beneficial to use larger batches. THe issues is that this might be worst for other parts of the networks. 
# SEED: here the seed is not optimized over because we are using random sampling (+ anyways it's ok to optimize over the initialization as long as it's done the same way for baselines also)
# the only different parameters with IVAE are the last 2 ones (all other hyperameters you can play around with them but I found that the current values were good)

# PREDICTOR
kwargs_multi="
$kwargs_multi
architecture@predictor=mlp_probe
+data_pred.kwargs.batch_size=64
optimizer_pred.kwargs.weight_decay=1e-5
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=plateau_quick
predictor.arch_kwargs.dropout_p=0.3
optimizer@optimizer_pred=Adam
featurizer.is_on_the_fly=false
" 



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "distortion=ince_basic" #"distortion=ince"        
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs_dep $kwargs $kwargs_multi  -m &

    sleep 7

    
  done
fi

wait



col_val_subset=""
rate_cols="['test/comm/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss', 'test/pred/loss']"
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