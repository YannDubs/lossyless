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
logger=none
optimizer@optimizer_feat=adam1e-4
checkpoint@checkpoint_feat=bestTrainLoss
scheduler@scheduler_feat=expdecay
data@data_feat=simclr_imagenet
evaluation.is_est_entropies=False
+data@data_pred=imagenet
data_pred.kwargs.dataset_kwargs.equivalence=[simclr_finetune]
optimizer@optimizer_pred=sslfinetuner
architecture@predictor=mlp1024
callbacks.is_force_no_additional_callback=true
data_pred.kwargs.batch_size=128
data_feat.kwargs.batch_size=128
+update_trainer_pred.max_epochs=2
trainer.max_epochs=4
scheduler@scheduler_pred=multistep
scheduler_pred.kwargs.MultiStepLR.milestones=[1,2,3]
trainer.val_check_interval=0.5
+mode=dev
trainer.limit_val_batches=0.1
trainer.limit_train_batches=0.1
trainer.limit_test_batches=1.0
featurizer.loss.beta=0.001
hydra.launcher.partition=rtx6000
$add_kwargs
"
#
#scheduler@scheduler_pred=cosine
# trainer.max_epochs=20 is because little time
# checkpoint@checkpoint_feat=bestTrainLoss because you are not augmenting validation set => only consider train

# every arguments that you are sweeping over
kwargs_multi="
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "featurizer=bottleneck_simclr,bottleneck_simclr_disjoint,bottleneck_simclr_postproj,simclr_lossless,bottleneck_simclr_nowarmup,bottleneck_simclr_noanneal,bottleneck_simclr_reinit trainer.gpus=4 +trainer.accelerator=ddp_spawn" "featurizer=bottleneck_simclr trainer.gpus=4" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi
