#!/usr/bin/env bash

experiment="hyp_clip_finetune"
notes="
**Goal**: Test and tune hyperaparmeters for finetuning clip
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
# TODO test the new pipeline for pretraiend features + add 
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
encoder.z_dim=512
is_only_feat=True
architecture@predictor=mlp_probe
data@data_feat=imagenet_clip
checkpoint@checkpoint_feat=bestTrainLoss
evaluation.is_est_entropies=False
trainer.val_check_interval=0.2
trainer.max_epochs=7
+update_trainer_pred.max_epochs=100
data_feat.kwargs.batch_size=128
featurizer=bottleneck_clip
finetune=clip
data_feat.kwargs.num_workers=4
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hypopt=multi_optuna
hydra.sweeper.optuna_config.n_trials=225
hydra.sweeper.optuna_config.n_jobs=50
monitor_direction=[minimize,minimize]
monitor_return=[test/feat/online_err,test/feat/rate]
hydra.sweeper.optuna_config.sampler=random
hydra.launcher.partition=rtx6000
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=tag(log,int(interval(64,128)))
featurizer.loss.beta_anneal=linear,constant
featurizer.loss.beta=tag(log,interval(1e-8,1e-2))
distortion.factor_beta=tag(log,interval(1e-5,1e-1))
rate.kwargs.warmup_k_epoch=int(interval(0,2))
rate.kwargs.is_endToEnd=true,false
rate.kwargs.invertible_processing=diag,psd
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-6,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-6,1e-4))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-5,3e-3))
optimizer@optimizer_online=SGD_likeadam,Adam
optimizer_online.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_online.kwargs.lr=tag(log,interval(1e-6,3e-3))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,3e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,unifmultistep100
scheduler@scheduler_coder=cosine,expdecay1000,plateau_quick,unifmultistep1000,unifmultistep100
scheduler@scheduler_online=cosine,expdecay100,expdecay1000,plateau_quick,unifmultistep1000,unifmultistep100
scheduler@scheduler_pred=cosine_restart,cosine,expdecay100,expdecay1000,plateau_quick,unifmultistep1000,unifmultistep100
seed=0,1,2,3,4
distortion.kwargs.project_kwargs.out_shape=tag(log,interval(0.01,0.5))
distortion.kwargs.temperature=tag(log,interval(0.005,0.1))
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi
