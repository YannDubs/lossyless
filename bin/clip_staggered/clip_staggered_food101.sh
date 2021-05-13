#!/usr/bin/env bash

experiment="clip_staggered_food101"
notes="
**Goal**: Test and tune MLP probe of FOOD101 with staggered CLIP
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=clip_staggered
experiment=$experiment 
timeout=$time
encoder.z_dim=512
is_only_feat=False
data@data_feat=coco
data@data_pred=food101
trainer.max_epochs=100
featurizer=bottleneck_clip_lossyZ
featurizer.is_on_the_fly=false
data_feat.kwargs.num_workers=4
architecture@predictor=mlp_probe
checkpoint@checkpoint_pred=bestValLoss
paths.pretrained.load=$pretrained_path
featurizer.is_train=false
evaluation.communication.ckpt_path=null
rate.kwargs.side_z_dim=128
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=50
hydra.sweeper.n_jobs=25
monitor_direction=[minimize]
monitor_return=[test/pred/err]
"

# PREDICTOR
# parameters for the predictor
kwargs_multi="
$kwargs_hypopt
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=Adam,SGD_likeadam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,1e-3))
scheduler@scheduler_pred=plateau_quick,unifmultistep1000,cosine_restart,cosine,expdecay100,expdecay1000,plateau
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
seed=int(interval(0,10))
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
       $col_val_subset \
       agg_mode=[summarize_metrics]