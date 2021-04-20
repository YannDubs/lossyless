#!/usr/bin/env bash

experiment="hyp_clip_lossyZ_pcam"
notes="
**Goal**: Test and tune hyperaparmeters for staggered clip
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
featurizer=bottleneck_clip_lossyZ
data_feat.kwargs.num_workers=4
featurizer.is_on_the_fly=false
optimizer@optimizer_pred=Adam
featurizer.loss.beta_anneal=linear
rate.kwargs.invertible_processing=diag
data_feat.kwargs.batch_size=32
scheduler@scheduler_coder=expdecay1000
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=1
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
featurizer.loss.beta=tag(log,interval(5e-2,1e-1))
distortion.factor_beta=tag(log,interval(1e-4,5e-2))
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(5e-6,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(5e-5,7e-4))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-6,7e-6))
optimizer_coder.kwargs.lr=tag(log,interval(1e-5,5e-4))
scheduler@scheduler_feat=cosine,plateau_quick
seed=0,1,2,3,4,5,6,7,8,9
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-5,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(2e-5,3e-4))
scheduler@scheduler_pred=plateau_quick,unifmultistep1000
predictor.arch_kwargs.dropout_p=interval(0.4,0.5)
" 


# kwargs_multi="
# trainer.max_epochs=1
# ++update_trainer_pred.max_epochs=5
# mode=dev
# +update_trainer_pred.limit_val_batches=1.0
# +update_trainer_pred.limit_test_batches=0.5
# evaluation.featurizer.is_evaluate=False
# data_feat.kwargs.batch_size=64
# data_pred.kwargs.batch_size=64
# "

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi
