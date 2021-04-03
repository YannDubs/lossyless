#!/usr/bin/env bash

experiment="hyp_cifar10_augmentations"
notes="
**Goal**: Understand which augmentations to apply for stl 10
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
is_only_feat=False
featurizer=neural_feat
architecture@encoder=resnet18
architecture@predictor=mlp_probe
data@data_feat=cifar10_aug
data@data_pred=cifar10_aug
rate=H_hyper
trainer.max_epochs=100
+update_trainer_pred.max_epochs=100
checkpoint@checkpoint_feat=bestTrainLoss
$add_kwargs
"


# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
distortion=ince,ivae
featurizer.loss.beta=1e-5,1e-6,1e-4
" 
#vae,nce

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "data_feat.kwargs.dataset_kwargs.equivalence=[hflip,resize_crop,auto_imagenet] data_pred.kwargs.dataset_kwargs.equivalence=[hflip,resize_crop,auto_imagenet]" # "data_feat.kwargs.dataset_kwargs.equivalence=[auto_cifar10] data_pred.kwargs.dataset_kwargs.equivalence=[auto_cifar10]" "data_feat.kwargs.dataset_kwargs.equivalence=[auto_imagenet] data_pred.kwargs.dataset_kwargs.equivalence=[auto_imagenet]" "data_feat.kwargs.dataset_kwargs.equivalence=[hflip,resize_crop,color,gray] data_pred.kwargs.dataset_kwargs.equivalence=[hflip,resize_crop,color,gray]"         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi

wait

col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_err','test/pred/loss','test/pred/err','train/pred/err']"
compare="dist"
data="merged" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       +summarize_RD_curves.data="${data}" \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="[]" \
       +plot_all_RD_curves.data="${data}" \
       +plot_all_RD_curves.rate_cols="${rate_cols}" \
       +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
       +plot_all_RD_curves.hue=$compare \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves] 

# !if you want an additional parameter from the configs use something like:
# +collect_data.kwargs.params_to_add.lr="optimizer_feat.kwargs.lr" \