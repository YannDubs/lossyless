#!/usr/bin/env bash

experiment="augmnist_stag_step1"
notes="
**Goal**: Understand effect of using staggered VS end to end for compression
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
pretrained_path="$SCRIPTPATH"/../../pretrained

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
is_only_feat=True
featurizer=neural_feat
architecture@encoder=resnet18
data@data_pred=mnist_aug
checkpoint@checkpoint_feat=bestTrainLoss
trainer.max_epochs=100
featurizer.loss.beta=1
distortion.factor_beta=1
rate.factor_beta=1
rate=lossless
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
data@data_feat=mnist_aug
seed=1
" 


if [ "$is_plot_only" = false ] ; then
  for dist in  "VIC" "BINCE"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi distortion=$dist paths.pretrained.staggered=$pretrained_path/lossless/$dist -m &

    sleep 3
    
  done
fi




