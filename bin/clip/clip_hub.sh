#!/usr/bin/env bash

experiment="clip_hub"
notes="
**Goal**: Save all pretrained (factorized) models to pytorch hub.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
pretrained_path="$SCRIPTPATH"/../../hub

kwargs="
experiment=$experiment 
timeout=$time
encoder.z_dim=512
is_only_feat=True
data@data_feat=coco
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=30
featurizer=bottleneck_clip_lossyZ_factorized
$add_kwargs
"

# TRAIN FACTORIZED MODEL
kwargs_multi="" 

if [ "$is_plot_only" = false ] ; then
  for beta in  "1e-01"   "5e-02"    "1e-02"          
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi featurizer.loss.beta=$beta paths.pretrained.save=$pretrained_path/beta$beta -m &

    sleep 3

  done
fi

wait


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
encoder.z_dim=512
data@data_feat=coco
featurizer=bottleneck_clip_lossyZ_factorized
checkpoint@checkpoint_pred=bestValLoss
featurizer.is_train=false
evaluation.communication.ckpt_path=null
$add_kwargs
"

for beta in  "1e-01"   "5e-02"    "1e-02"          
  do

    col_val_subset=""
       python utils/save_hub.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       featurizer.loss.beta=$beta \
       paths.pretrained.load=$pretrained_path/beta$beta \
       server=local \
       trainer.gpus=0 \
       load_pretrained.mode=[] 

  done

