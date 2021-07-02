#!/usr/bin/env bash

experiment="hyp_text_stl10"
notes="
**Goal**: Understand how ince with text works
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
architecture@encoder=resnet18
architecture@predictor=mlp_probe
logger.kwargs.project=hypopt
is_only_feat=False
featurizer=neural_feat
data@data_feat=coco
data@data_pred=stl10
checkpoint@checkpoint_feat=bestValLoss
checkpoint@checkpoint_pred=bestValLoss
rate=H_hyper
trainer.max_epochs=50
featurizer=neural_feat
featurizer.is_on_the_fly=false
data_feat.kwargs.batch_size=128
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
monitor_return=[test/feat/distortion,test/feat/rate]
"

# here the random sampler means that we are not actually doing smart hyperparametr tuning use `nsgaii` if you want
# n_trials is total trials and `n_jobs` are the ones to run in batch (when random it only changes compute but it's important when using other samplers)
# monitor is what you want to optimize ( only useful for plotting when using random sampler, but improtant otherwise)

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
featurizer.loss.beta=tag(log,interval(3e-4,5e-2))
distortion.factor_beta=tag(log,interval(3e-4,1e-2))
seed=0,1,2,3,4,5,6,7,8,9
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=3e-4
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=1e-6
optimizer_coder.kwargs.lr=1e-4
scheduler@scheduler_feat=expdecay100
scheduler@scheduler_coder=expdecay100
rate.kwargs.warmup_k_epoch=0,1
distortion=ince_text,ince_text_basic
distortion.kwargs.temperature=0.01,0.03,0.1
distortion.kwargs.is_batch_neg=true,false
distortion.kwargs.is_train_temperature=true,false
" 
# distortion.factor_beta : instead of deacreasing weight given to rate will increase weight given to distortion
# BATCH SIZE: for INCE it can be beneficial to use larger batches. THe issues is that this might be worst for other parts of the networks. 
# SEED: here the seed is not optimized over because we are using random sampling (+ anyways it's ok to optimize over the initialization as long as it's done the same way for baselines also)
# the only different parameters with IVAE are the last 2 ones (all other hyperameters you can play around with them but I found that the current values were good)

kwargs_multi="
$kwargs_multi
architecture@predictor=mlp_probe
data_pred.kwargs.batch_size=64
optimizer_pred.kwargs.weight_decay=1e-5
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=plateau_quick
predictor.arch_kwargs.dropout_p=0.3
optimizer@optimizer_pred=Adam
featurizer.is_on_the_fly=false
" 


# kwargs_multi="
# $kwargs_multi
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
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_err','test/pred/loss','test/pred/err']"
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
