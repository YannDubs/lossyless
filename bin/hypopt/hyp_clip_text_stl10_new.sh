#!/usr/bin/env bash

experiment="hyp_clip_text_stl10_new"
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
checkpoint@checkpoint_pred=bestValLoss
featurizer.is_on_the_fly=false
trainer.max_epochs=50
+update_trainer_pred.max_epochs=100
featurizer=bottleneck_clip
data_feat.kwargs.num_workers=4
data_feat.kwargs.batch_size=128
data_pred.kwargs.batch_size=64
$add_kwargs
"
# data_pred should not be augmented (it won't here due to no ot the fly but better if it was explicit)

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=50
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
hydra.launcher.partition=rtx6000
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
featurizer.loss.beta=tag(log,interval(1e-4,1e-3))
distortion.factor_beta=tag(log,interval(1e-4,1e-3))
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
rate.kwargs.invertible_processing=diag
distortion=ince_text_basic
distortion.kwargs.is_batch_neg=false
distortion.kwargs.is_train_temperature=false
finetune=clip
" 

# PREDICTOR
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


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi

wait 

