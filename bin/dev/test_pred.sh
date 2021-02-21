#!/usr/bin/env bash

experiment="test_pred"
notes="
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
is_only_feat=False
architecture@encoder=resnet18
architecture@predictor=resnet18
distortion.kwargs.arch_kwargs.complexity=3
encoder.z_dim=128
data@data_feat=augmnist
+data@data_pred=mnist
evaluation.is_est_entropies=False
rate=H_hyper
optimizer@optimizer_pred=sgd 
scheduler@scheduler_pred=multistep
optimizer_pred.is_lr_find=False
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
featurizer=neural_rec
distortion=ivae
trainer.max_epochs=1
+update_trainer_pred.max_epochs=30
" 
# seed=1,2,3,4,5

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi

