#!/usr/bin/env bash

experiment="bottleneck_mlp_eval"
notes="
**Goal**: MLP evaluation of different pretrained resnet50.
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
data@data_feat=coco
trainer.max_epochs=100
featurizer.is_on_the_fly=false
data_feat.kwargs.num_workers=4
architecture@predictor=mlp_probe
checkpoint@checkpoint_pred=bestValLoss
featurizer.is_train=false
evaluation.communication.ckpt_path=null
evaluation.predictor.is_eval_train=True
$add_kwargs
"

# PREDICTOR
# parameters for the predictor
kwargs_multi="
optimizer@optimizer_pred=AdamW
optimizer_pred.kwargs.weight_decay=1e-7
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=expdecay1000
predictor.arch_kwargs.dropout_p=0
data@data_pred=stl10,imagenet,cars196,caltech101,food101,pcam,pets37,cifar10
++data_pred.kwargs.batch_size=64
" 

# PREDICTOR
# parameters for the predictor
kwargs_multi="
optimizer@optimizer_pred=AdamW
optimizer_pred.kwargs.weight_decay=1e-7
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=expdecay1000
predictor.arch_kwargs.dropout_p=0
data@data_pred=cifar100
++data_pred.kwargs.batch_size=64
" 


if [ "$is_plot_only" = false ] ; then
  for beta in  "3e-02" "1e-01" # "1e-03" "3e-03" #"3e-02" "1e-01"  "1e+00"  "1e+01"  #"3e-02" "3e-01" "3e+00"
  do
    for model in   "sup"  # "simclr" #"swav"   #"clip"
    do
    python "$main"  +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi  featurizer.loss.beta=$beta paths.pretrained.load=$pretrained_path/rn50"$model"/beta"$beta" featurizer=bottleneck_rn50"$model"_lossyZ -m &

    sleep 10
    done
  done
fi

wait

col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_err','test/pred/loss','test/pred/err']"
compare="datafeat"
data="merged" # want to access both ther featurizer data and the  predictor data
python utils/aggregate.py \
      experiment=$experiment  \
      $col_val_subset \
      patterns.featurizer=null \
      agg_mode=[summarize_metrics]  