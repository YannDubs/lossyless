#!/usr/bin/env bash

experiment="clip_stag_sweep"
notes="
**Goal**: Add an entropy bottleneck to CLIP, and trains only entropy bottleneck.This pretrains the generic compressor that will be reused for all downstream datasets.
"

pretrained_path=`dirname $0`/../../hub

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=clip_staggered
experiment=$experiment 
timeout=$time
encoder.z_dim=512
is_only_feat=True
data@data_feat=coco
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=50
featurizer=bottleneck_clip_lossyZ
optimizer_feat.kwargs.lr=1e-3
featurizer.loss.beta=5e-2
paths.pretrained.save=$pretrained_path
$add_kwargs
"

kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=10
hydra.sweeper.n_jobs=10
monitor_direction=[minimize]
monitor_return=[test/feat/rate]
trainer.max_epochs=10,20,30,40,50
optimizer_feat.kwargs.lr=tag(log,interval(1e-5,1e-3))
rate=H_hyper
+rate.kwargs.side_z_dim=16,32,64,128,256
scheduler@scheduler_feat=expdecay1000,unifmultistep1000,unifmultistep10000
scheduler@scheduler_coder=expdecay1000,unifmultistep1000,unifmultistep10000
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
