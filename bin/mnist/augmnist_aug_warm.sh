#!/usr/bin/env bash

experiment="augmnist_aug_warm"
notes="
**Goal**: Understand the impact of the choice of augmentation on a simple setting,
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
is_only_feat=False
featurizer=neural_feat
architecture@encoder=resnet18
data@data_pred=mnist_aug
checkpoint@checkpoint_feat=bestTrainLoss
rate=H_hyper
trainer.max_epochs=100
rate.kwargs.warmup_k_epoch=5
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
data@data_feat=mnist_aug
seed=1
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "distortion=VIC featurizer.loss.beta=1e-5,1e-4,1e-3,1e-2,0.1,1" "distortion=BINCE featurizer.loss.beta=1e-7,1e-6,1e-5,1e-4,1e-3,1e-2"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
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
      +plot_all_RD_curves.folder_col="dist" \
      +plot_all_RD_curves.data="${data}" \
      +plot_all_RD_curves.rate_cols="${rate_cols}" \
      +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
      +plot_all_RD_curves.hue=$compare \
      +plot_all_RD_curves.plot_config_kwargs.font_scale=1 \
      +summarize_RD_curves.data="${data}" \
      +summarize_RD_curves.rate_cols="${rate_cols}" \
      +summarize_RD_curves.distortion_cols="${distortion_cols}" \
      +summarize_RD_curves.mse_cols="${distortion_cols}" \
      agg_mode=[summarize_metrics,plot_all_RD_curves,summarize_RD_curves]  


col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/pred/err']"
compare="datafeat"
data="merged" # want to access both ther featurizer data and the  predictor data
python utils/aggregate.py \
      experiment=$experiment  \
      $col_val_subset \
      kwargs.prfx="single" \
      +plot_all_RD_curves.folder_col="dist" \
      +plot_all_RD_curves.data="${data}" \
      +plot_all_RD_curves.rate_cols="${rate_cols}" \
      +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
      +plot_all_RD_curves.hue=$compare \
      +plot_all_RD_curves.plot_config_kwargs.font_scale=1 \
      agg_mode=[plot_all_RD_curves]  