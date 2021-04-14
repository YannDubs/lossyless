#!/usr/bin/env bash

experiment="hyp_clip_lossyZ_flowers102"
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
data@data_pred=flowers102
checkpoint@checkpoint_feat=bestValLoss
evaluation.is_est_entropies=False
featurizer.is_on_the_fly=false
trainer.max_epochs=30
+update_trainer_pred.max_epochs=100
featurizer=bottleneck_clip_lossyZ
data_feat.kwargs.num_workers=4
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=225
hydra.sweeper.n_jobs=50
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=tag(log,int(interval(16,128)))
data_pred.kwargs.batch_size=tag(log,int(interval(16,64)))
featurizer.loss.beta_anneal=linear,constant
featurizer.loss.beta=tag(log,interval(1e-5,1e0))
distortion.factor_beta=tag(log,interval(1e-5,1e0))
rate.kwargs.invertible_processing=diag,psd
optimizer@optimizer_feat=Adam,AdamW,SGD_likeadam
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-5,1e-2))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-5,1e-2))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,1e-2))
scheduler@scheduler_feat=cosine,expdecay1000,plateau_quick,unifmultistep100
scheduler@scheduler_coder=cosine,expdecay1000,plateau_quick,unifmultistep100
scheduler@scheduler_pred=cosine_restart,cosine,expdecay1000,plateau_quick,unifmultistep1000
seed=0,1,2,3,4
" 


kwargs_multi="
trainer.max_epochs=1
++update_trainer_pred.max_epochs=5
mode=dev
+update_trainer_pred.limit_val_batches=1.0
+update_trainer_pred.limit_test_batches=0.5
evaluation.featurizer.is_evaluate=False
data_feat.kwargs.batch_size=64
data_pred.kwargs.batch_size=64
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep #-m &

    sleep 7

    
  done
fi
