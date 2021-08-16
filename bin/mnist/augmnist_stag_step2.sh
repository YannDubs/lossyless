#!/usr/bin/env bash

experiment="augmnist_stag_step2"
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
is_only_feat=False
featurizer=neural_feat
architecture@encoder=resnet18
data@data_pred=mnist_aug
checkpoint@checkpoint_feat=bestTrainLoss
trainer.max_epochs=50
+update_trainer_pred.max_epochs=100
featurizer.loss.beta=1
distortion.factor_beta=1
rate.factor_beta=1
rate=H_hyper
rate.kwargs.is_endToEnd=False
finetune=freezer
distortion=lossy_Z
scheduler@scheduler_feat=unifmultistep1000
scheduler@scheduler_coder=unifmultistep1000
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
data@data_feat=mnist_aug
seed=1
featurizer.loss.beta=1e-2
" 
#VIC
# beta should now have no impact. What is important is the weight


if [ "$is_plot_only" = false ] ; then
  for dist in  "VIC" "BINCE"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi  paths.pretrained.staggered=$pretrained_path/lossless/$dist -m &

    sleep 3
    
  done
fi




