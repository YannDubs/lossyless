#!/usr/bin/env bash

experiment="imagenet_simclr_workshop"
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
optimizer@optimizer_feat=adam1e-6
optimizer_feat.lr_rate_factor=100
checkpoint@checkpoint_feat=bestTrainLoss
scheduler@scheduler_feat=expdecay
data@data_feat=simclr_imagenet
evaluation.is_est_entropies=False
+data@data_pred=imagenet
data_pred.kwargs.dataset_kwargs.equivalence=[simclr_finetune]
optimizer@optimizer_pred=sslfinetuner
architecture@predictor=mlp1024
$add_kwargs
"
#
#scheduler@scheduler_pred=cosine
# trainer.max_epochs=20 is because little time
# checkpoint@checkpoint_feat=bestTrainLoss because you are not augmenting validation set => only consider train

# every arguments that you are sweeping over
kwargs_multi="
seed=1
data_pred.kwargs.batch_size=128
data_feat.kwargs.batch_size=128
hydra.launcher.partition=rtx6000
+update_trainer_pred.max_epochs=30
trainer.max_epochs=20
scheduler@scheduler_pred=multistep
scheduler_pred.kwargs.MultiStepLR.milestones=[17,25]
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in    "featurizer=simclr encoder.z_dim=2048" "featurizer=bottlenecksimclr featurizer.loss.beta=0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m & 

    sleep 3
    
  done
fi
