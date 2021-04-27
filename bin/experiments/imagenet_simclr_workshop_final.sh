#!/usr/bin/env bash

experiment="imagenet_simclr_workshop_final"
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
encoder.z_dim=2048
optimizer@optimizer_feat=adam1e-6
optimizer_feat.lr_rate_factor=100
checkpoint@checkpoint_feat=bestTrainLoss
scheduler@scheduler_feat=expdecay
data@data_feat=simclr_imagenet
trainer.val_check_interval=0.25
$add_kwargs
"
#
#scheduler@scheduler_pred=cosine
# trainer.max_epochs=20 is because little time
# checkpoint@checkpoint_feat=bestTrainLoss because you are not augmenting validation set => only consider train
# trainer.val_check_interval=0.25 checkpoints every 1/4th of epoch

# every arguments that you are sweeping over
kwargs_multi="
seed=1
data_feat.kwargs.batch_size=128
hydra.launcher.partition=rtx6000
trainer.max_epochs=5
rate=H_factorized
featurizer=simclr_final
is_only_feat=True
featurizer.loss.beta=1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e-1,1
rate.kwargs.init_scale=10
" 

kwargs_multi="
seed=1
data_feat.kwargs.batch_size=32
trainer.max_epochs=5
featurizer=simclr_final
is_only_feat=True
featurizer.loss.beta=1e-8
rate.kwargs.init_scale=10
trainer.limit_val_batches=0.005
trainer.limit_train_batches=0.0005
trainer.limit_test_batches=0.005
trainer.val_check_interval=1.0
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep 

    sleep 3
    
  done
fi
