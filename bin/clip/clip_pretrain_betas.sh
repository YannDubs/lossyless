#!/usr/bin/env bash

experiment="clip_pretrain_betas"
notes="
**Goal**: Pretrain compressors for different values of beta.
"


wait


# parses special mode for running the script
source `dirname $0`/../utils.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
pretrained_path="$SCRIPTPATH"/../../hub

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=clip_staggered
experiment=$experiment 
timeout=$time
encoder.z_dim=512
is_only_feat=True
data@data_feat=coco
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=30
featurizer=bottleneck_clip_lossyZ
$add_kwargs
"

kwargs_multi="" 

if [ "$is_plot_only" = false ] ; then
  for beta in  "1e-1"   #"5e-2"    "1e-2"          
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi featurizer.loss.beta=$beta paths.pretrained.save=$pretrained_path/beta$beta -m &

    sleep 3

  done
fi

wait



