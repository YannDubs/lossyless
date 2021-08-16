#!/usr/bin/env bash

experiment="bottleneck_pretrain"
notes="
**Goal**: Pretrain compressors for different values of beta.
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
data@data_feat=coco
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=30
$add_kwargs
"

kwargs_multi="" 

if [ "$is_plot_only" = false ] ; then
  for beta in  "1e-03" "3e-03"  # "1e-02" "3e-02" "1e-01" "3e-01" "1e+00" "3e+00" "1e+01"
  do
    for model in    "clip" # "sup" "clip" "swav" "simclr"
    do

      python "$main"  +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi featurizer.loss.beta="$beta" paths.pretrained.save=$pretrained_path/rn50"$model"/beta"$beta" featurizer=bottleneck_rn50"$model"_lossyZ -m &

      sleep 3
    done

  done
fi

wait



