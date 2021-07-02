#!/usr/bin/env bash

experiment="hyp_ivae_stl10"
notes="
**Goal**: Hyperparameter tuning for ivae on stl10
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
trainer.max_epochs=50
+update_trainer_pred.max_epochs=100
distortion=ivae
$add_kwargs
"
# use many more epochs for the predictor because there's little data availabe
#TODO increase epochs before pushing

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

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=tag(log,int(interval(32,128)))
encoder.z_dim=tag(log,int(interval(32,512)))
featurizer.loss.beta_anneal=linear,constant
featurizer.loss.beta=tag(log,interval(1e-8,1e0))
distortion.factor_beta=tag(log,interval(1e-5,100))
rate.kwargs.warmup_k_epoch=int(interval(0,3))
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer_feat.kwargs.is_lars=true,false
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,plateau
scheduler@scheduler_coder=cosine_restart,expdecay100,plateau_quick,unifmultistep100
seed=0,1,2,3,4
" 
# distortion.factor_beta : instead of deacreasing weight given to rate will increase weight given to distortion
# SEED: here the seed is not optimized over because we are using random sampling (+ anyways it's ok to optimize over the initialization as long as it's done the same way for baselines also)

# PREDICTOR
kwargs_multi="
$kwargs_multi
data_pred.kwargs.batch_size=tag(log,int(interval(16,64)))
predictor.arch_kwargs.hid_dim=tag(log,int(interval(1024,4096)))
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
predictor.arch_kwargs.n_hid_layers=1,2
optimizer@optimizer_pred=SGD_likeadam,Adam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,expdecay1000,unifmultistep100
" 


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