#!/usr/bin/env bash

experiment="hyp_simclr"
notes="
**Goal**: Pretrained SimCLR model
"

# e.g. command: bin/experiments/cifar10_RD_compare_rates.sh -s vector -t 1440

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
is_only_feat=True
checkpoint@checkpoint_feat=bestTrainLoss
optimizer@optimizer_feat=adam1e-4
scheduler@scheduler_feat=expdecay
data@data_feat=simclr_imagenet
evaluation.is_est_entropies=False
data_feat.kwargs.batch_size=128
trainer.max_epochs=4
trainer.val_check_interval=0.1
featurizer.loss.beta=0.00001
featurizer=bottleneck_simclr
callbacks.LearningRateMonitor.is_use=False
$add_kwargs
"
#scheduler@scheduler_pred=cosine
# trainer.max_epochs=20 is because little time
# checkpoint@checkpoint_feat=bestTrainLoss because you are not augmenting validation set => only consider train

# every arguments that you are sweeping over
kwargs_multi="
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "featurizer=bottleneck_simclr_disjoint,simclr_freeze" "optimizer_feat.kwargs.is_lars=true" "optimizer@optimizer_feat=wadam1e-6,wadam1e-5,wadam1e-7,wadam1e-4" "featurizer.loss.beta=1e-7,1e-6,1e-5,1e-4,1e-3,1e-2"   "featurizer=bottleneck_simclr_disjoint" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7
    
  done
fi
