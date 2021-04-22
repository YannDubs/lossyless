#!/usr/bin/env bash

experiment="clip_staggered_galaxy"
notes="
**Goal**: Test and tune MLP probe of Galaxy with staggered CLIP
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=clip_staggered
experiment=$experiment 
timeout=$time
encoder.z_dim=512
is_only_feat=True
architecture@predictor=mlp_probe
data@data_feat=coco
data@data_pred=galaxy
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=100
featurizer=bottleneck_clip_lossyZ
data_feat.kwargs.num_workers=4
featurizer.is_on_the_fly=false
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=30
hydra.sweeper.n_jobs=10
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=32
featurizer.loss.beta=5e-2
distortion.factor_beta=1e-3
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=3e-4
scheduler@scheduler_feat=plateau_quick
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=3e-6
optimizer_coder.kwargs.lr=1e-4
scheduler@scheduler_coder=expdecay100
" 

kwargs_multi="
$kwargs_hypopt
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=Adam,SGD_likeadam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-6,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,1e-3))
scheduler@scheduler_pred=plateau_quick,unifmultistep1000,cosine_restart
predictor.arch_kwargs.dropout_p=interval(0.2,0.5)
seed=0,1,2,3,4,5,6,7,8,9
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

  done
fi

wait 

# for featurizer
col_val_subset=""
python aggregate.py \
       experiment=$experiment  \
       patterns.predictor=null \
       $col_val_subset \
       agg_mode=[summarize_metrics]