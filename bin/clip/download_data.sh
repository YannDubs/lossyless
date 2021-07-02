#!/usr/bin/env bash

experiment="download_data"
notes="
**Goal**: download all data
"

pip install tensorflow-datasets -U

wait

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger=none
encoder.z_dim=512
+update_trainer_feat.max_epochs=1
+update_trainer_feat.limit_train_batches=1
featurizer=clip_freeze
is_only_feat=True
data_feat.kwargs.num_workers=4
"


kwargs_multi="" 

if [ "$is_plot_only" = false ] ; then
  for data in "coco" "stl10" "cifar10" "caltech101" "cars196" "food101" "imagenet" "pcam" "pets37" "cifar100"        
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi data@data_feat=$data  -m 

    wait

  done
fi