#!/usr/bin/env bash

experiment="dev_hyp_clip"
notes="
**Goal**: Pretrained CLIP model
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
data_feat.kwargs.batch_size=32
featurizer.loss.beta=0.00001
featurizer=clip_freeze
finetune=clip
trainer.max_epochs=1
trainer.val_check_interval=1.0
+mode=dev
trainer.limit_val_batches=0.005
trainer.limit_train_batches=0.00005
trainer.limit_test_batches=0.005
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
  for kwargs_dep in  "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep 

    sleep 7
    
  done
fi
