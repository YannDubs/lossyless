#!/usr/bin/env bash

experiment="hyp_clip_finetune_tmp"
notes="
**Goal**: Test and tune hyperaparmeters for finetuning clip
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
data@data_feat=imagenet_clip
checkpoint@checkpoint_feat=bestTrainLoss
evaluation.is_est_entropies=False
trainer.val_check_interval=0.2
featurizer.is_on_the_fly=false
trainer.max_epochs=5
+update_trainer_pred.max_epochs=50
+update_trainer_pred.val_check_interval=1.0
featurizer=bottleneck_clip_img
finetune=clip
data_feat.kwargs.num_workers=4
rate.kwargs.invertible_processing=diag
optimizer@optimizer_feat=AdamW
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hypopt=multi_optuna
hydra.sweeper.optuna_config.n_trials=225
hydra.sweeper.optuna_config.n_jobs=50
monitor_direction=[minimize,minimize]
monitor_return=['test/pred/err','test/feat/rate']
hydra.sweeper.optuna_config.sampler=random
hydra.launcher.partition=rtx6000
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=tag(log,int(interval(64,128)))
data_pred.kwargs.batch_size=tag(log,int(interval(16,64)))
featurizer.loss.beta_anneal=linear,constant
featurizer.loss.beta=tag(log,interval(1e-6,1e-3))
distortion.factor_beta=tag(log,interval(1e-5,1e-1))
rate.kwargs.warmup_k_epoch=int(interval(1,2))
rate.kwargs.is_endToEnd=true,false
optimizer_feat.kwargs.weight_decay=tag(log,interval(5e-6,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-6,5e-5))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-5,3e-3))
optimizer@optimizer_online=SGD_likeadam,Adam
optimizer_online.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_online.kwargs.lr=tag(log,interval(1e-4,5e-3))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,3e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,unifmultistep100
scheduler@scheduler_coder=cosine,expdecay1000,plateau_quick,unifmultistep100
scheduler@scheduler_online=cosine,expdecay100,expdecay1000,plateau_quick,unifmultistep100
scheduler@scheduler_pred=cosine_restart,cosine,expdecay100,expdecay1000,plateau_quick,unifmultistep1000,unifmultistep100
seed=1,2,3,4,5
distortion.kwargs.temperature=tag(log,interval(0.005,0.07))
distortion.kwargs.is_train_temperature=false,true
distortion.kwargs.project_kwargs.out_shape=tag(log,interval(0.01,0.5))
" 


# kwargs_multi="
# $kwargs_multi
# hydra.sweeper.optuna_config.n_jobs=1
# +trainer.limit_val_batches=0.1
# +trainer.limit_test_batches=0.1
# +trainer.limit_train_batches=0.05
# trainer.max_epochs=1
# trainer.val_check_interval=0.5
# "

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi