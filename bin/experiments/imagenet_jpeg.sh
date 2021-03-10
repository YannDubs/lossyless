#!/usr/bin/env bash

experiment="imagenet_jpeg"
notes=""

# e.g. command: bin/experiments/cifar10_RD_compare_rates.sh -s vector -t 1440

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
data@data_feat=imagenet
evaluation.is_est_entropies=False
trainer.max_epochs=3
optimizer@optimizer_pred=pretrained
scheduler@scheduler_pred=expdecay
featurizer=jpeg--
seed=1
$add_kwargs
"
# +trainer.update_trainer_feat.max_epochs=10 is because little time

# every arguments that you are sweeping over
kwargs_multi="
+data_pred.kwargs.batch_size=32
data_feat.kwargs.batch_size=32
featurizer.quality=95
is_only_feat=True
evaluation.featurizer.is_evaluate=True
" 
#trainer.gpus=8

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi
