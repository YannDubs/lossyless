#!/usr/bin/env bash

experiment="hyp_clip_text_cifar100"
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
data@data_pred=cifar100
checkpoint@checkpoint_feat=bestValLoss
evaluation.is_est_entropies=False
featurizer.is_on_the_fly=false
trainer.max_epochs=30
+update_trainer_pred.max_epochs=150
featurizer=bottleneck_clip
data_feat.kwargs.num_workers=4
rate.kwargs.invertible_processing=diag
data_feat.kwargs.batch_size=128
rate.kwargs.warmup_k_epoch=1
distortion.kwargs.is_train_temperature=true
optimizer@optimizer_pred=Adam
optimizer@optimizer_feat=Adam
$add_kwargs
"
# data_pred should not be augmented (it won't here due to no ot the fly but better if it was explicit)

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=300
hydra.sweeper.n_jobs=150
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/feat/rate]
hydra.launcher.partition=rtx6000
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_pred.kwargs.batch_size=tag(log,int(interval(16,32)))
featurizer.loss.beta=tag(log,interval(1e-6,1e-3))
distortion.factor_beta=tag(log,interval(1e-4,1e-1))
rate.kwargs.is_endToEnd=true,false
featurizer.loss.beta_anneal=linear,constant
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-5,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(3e-6,3e-5))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-5,1e-3))
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-6,5e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,5e-4))
scheduler@scheduler_feat=expdecay100,plateau_quick
scheduler@scheduler_coder=expdecay1000,unifmultistep1000,cosine,plateau_quick
scheduler@scheduler_pred=cosine_restart,cosine,expdecay1000,plateau_quick
seed=0,1,2,3,4,5,6,7,8,9
distortion.kwargs.temperature=tag(log,interval(1e-3,3e-2))
predictor.arch_kwargs.dropout_p=interval(0.1,0.4)
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
