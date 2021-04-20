#!/usr/bin/env bash

experiment="imagenet_webp_workshop"
notes=""

# e.g. command: bin/experiments/cifar10_RD_compare_rates.sh -s vector -t 1440

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
data@data_feat=imagenet
trainer.max_epochs=3
optimizer@optimizer_pred=pretrained
scheduler@scheduler_pred=expdecay
architecture@predictor=resnet50
+predictor.arch_kwargs.is_pretrained=True
featurizer=webp++
seed=1
trainer.val_check_interval=0.25
$add_kwargs
"
# +trainer.update_trainer_feat.max_epochs=10 is because little time

# every arguments that you are sweeping over
kwargs_multi="
+data_pred.kwargs.batch_size=128
data_feat.kwargs.batch_size=128
hydra.launcher.partition=rtx6000
featurizer.quality=1,5,10,20,40,60,95
evaluation.featurizer.is_evaluate=False
" 
#trainer.gpus=8

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi
