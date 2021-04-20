#!/usr/bin/env bash

experiment="hyp_clip_text_stl10"
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
data@data_pred=stl10
checkpoint@checkpoint_feat=bestValLoss
featurizer.is_on_the_fly=false
trainer.max_epochs=50
+update_trainer_pred.max_epochs=150
featurizer=bottleneck_clip
data_feat.kwargs.num_workers=4
rate.kwargs.invertible_processing=diag
data_feat.kwargs.batch_size=128
rate.kwargs.warmup_k_epoch=1
distortion.kwargs.is_train_temperature=true
featurizer.loss.beta_anneal=linear
optimizer@optimizer_pred=Adam
optimizer@optimizer_feat=Adam
optimizer@optimizer_coder=Adam
$add_kwargs
"
# data_pred should not be augmented (it won't here due to no ot the fly but better if it was explicit)

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=20
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
hydra.launcher.partition=rtx6000
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
featurizer.loss.beta=tag(log,interval(1e-6,1e-3))
distortion.factor_beta=tag(log,interval(1e-4,1e-1))
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-5,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(3e-6,1e-4))
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-5,1e-3))
scheduler@scheduler_feat=expdecay100,plateau_quick
scheduler@scheduler_coder=expdecay1000,unifmultistep1000,cosine,plateau_quick
distortion.kwargs.temperature=tag(log,interval(1e-3,3e-2))
seed=0,1,2,3,4,5,6,7,8,9
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,3e-3))
scheduler@scheduler_pred=cosine_restart,plateau_quick,unifmultistep1000,expdecay1000
finetune=clip,clip_slow
" 

# kwargs_multi="
# trainer.val_check_interval=1.0
# trainer.max_epochs=1
# ++update_trainer_pred.max_epochs=5
# mode=dev
# is_only_feat=False
# featurizer.is_on_the_fly=false
# +update_trainer_pred.limit_val_batches=1.0
# +update_trainer_pred.limit_test_batches=0.5
# optimizer_feat.kwargs.lr=1e-3
# optimizer_pred.kwargs.lr=1e-3
# optimizer_coder.kwargs.lr=1e-3
# data_feat.kwargs.batch_size=64
# data_pred.kwargs.batch_size=64
# monitor_direction=[minimize,minimize]
# monitor_return=[test/pred/err,test/feat/rate]
# "

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi
