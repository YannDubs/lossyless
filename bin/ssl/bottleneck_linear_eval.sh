#!/usr/bin/env bash

experiment="bottleneck_linear_eval"
notes="
**Goal**: Linear evaluation of different pretrained resnet50.
"


# parses special mode for running the script
source `dirname $0`/../utils.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
pretrained_path="$SCRIPTPATH"/../../pretrained


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger=none
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


kwargs_multi="
data@data_pred=stl10,cars196,caltech101,food101,pcam,pets37,cifar10,cifar100
" 

if [ "$is_plot_only" = false ] ; then
  for beta in   "1e-01" # "1e-03" "3e-03" #"3e-02" "1e-01"  "1e+00"  "1e+01"  #"3e-02" "3e-01" "3e+00"
  do
    for model in   "sup"  # "simclr" #"swav"   #"clip"
    do
    python utils/Z_linear_eval.py $kwargs $kwargs_multi  featurizer.loss.beta=$beta paths.pretrained.load=$pretrained_path/rn50"$model"/beta"$beta" featurizer=bottleneck_rn50"$model"_lossyZ -m &

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