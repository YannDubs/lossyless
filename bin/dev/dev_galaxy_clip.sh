#!/usr/bin/env bash

experiment="dev_galaxy_clip" # should always be called with -m  dev
notes="
**Goal**: Checking that clip works
**Hypothesis**: No errors
"

source `dirname $0`/../utils.sh

# project and server kwargs
kwargs="
logger.kwargs.project=lossyless
wandb_entity=${env:USER}
experiment=$experiment
timeout=$time
$add_kwargs
"

# experiment kwargs
kwargs="
$kwargs
encoder.z_dim=512
data@data_feat=galaxy256
trainer.max_epochs=3
featurizer=bottleneck_clip_lossyZ
data_feat.kwargs.num_workers=4
architecture@predictor=mlp_probe
checkpoint@checkpoint_pred=bestValLoss
checkpoint@checkpoint_feat=bestTrainLoss
"

kwargs_multi="
featurizer.is_on_the_fly=false,true
$add_kwargs
"

# PREDICTOR
# parameters for the predictor
kwargs_multi="
$kwargs_hypopt
+data_pred.kwargs.batch_size=64
optimizer@optimizer_pred=AdamW
optimizer_pred.kwargs.weight_decay=1e-4
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=expdecay100
predictor.arch_kwargs.dropout_p=0.2
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do
    
    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi
