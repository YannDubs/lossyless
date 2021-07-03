#!/usr/bin/env bash

experiment="clip_bottleneck"
notes="
**Goal**: Add an entropy bottleneck to CLIP and evaluates on all data.
"



# parses special mode for running the script
source `dirname $0`/../utils.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
pretrained_path="$SCRIPTPATH"/../../hub


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=clip_staggered
experiment=$experiment 
timeout=$time
encoder.z_dim=512
data@data_feat=coco
trainer.max_epochs=100
featurizer=bottleneck_clip_lossyZ
featurizer.is_on_the_fly=false
data_feat.kwargs.num_workers=4
architecture@predictor=mlp_probe
checkpoint@checkpoint_pred=bestValLoss
featurizer.is_train=false
evaluation.communication.ckpt_path=null
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=10
hydra.sweeper.n_jobs=10
monitor_direction=[minimize]
monitor_return=[test/pred/loss]
"

# PREDICTOR
# parameters for the predictor
kwargs_multi="
$kwargs_hypopt
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=Adam,SGD_likeadam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,1e-3))
scheduler@scheduler_pred=plateau_quick,unifmultistep1000,cosine_restart,expdecay100,expdecay1000,plateau
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
seed=int(interval(0,10))
" 


if [ "$is_plot_only" = false ] ; then
  for data in "stl10" "imagenet" "cars196"  "caltech101"  "food101"  "pcam" "pets37" "cifar10"    "cifar100"        
  do
    for beta in     "1e-1"  "5e-2"   "1e-2"        
    do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi data@data_pred=$data featurizer.loss.beta=$beta paths.pretrained.load=$pretrained_path/beta$beta hydra.sweeper.study_name=$data_$beta -m &

    sleep 30

    done
    
  done
fi
