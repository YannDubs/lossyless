#!/usr/bin/env bash

experiment="clip_RD"
notes="
**Goal**: CLIP RD curve for imagenet
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=clip_staggered
experiment=$experiment 
timeout=$time
encoder.z_dim=512
data@data_feat=coco
data@data_pred=imagenet
trainer.max_epochs=20
+update_trainer_feat.max_epochs=20
trainer.max_epochs=100
featurizer=clip_freeze
featurizer.is_on_the_fly=false
data_feat.kwargs.num_workers=4
architecture@predictor=mlp_probe
checkpoint@checkpoint_pred=bestValLoss
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=50
hydra.sweeper.n_jobs=50
monitor_direction=[minimize]
monitor_return=[test/pred/loss]
"

# PREDICTOR
# parameters for the predictor
kwargs_multi="
$kwargs_hypopt
featurizer.loss.beta=tag(log,interval(1e-3,1e1))
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=Adam,SGD_likeadam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,1e-3))
scheduler@scheduler_pred=plateau_quick,unifmultistep1000,cosine_restart,expdecay100,expdecay1000,plateau
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
seed=int(interval(0,10))
" 



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi

wait 

# for featurizer
col_val_subset=""
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]