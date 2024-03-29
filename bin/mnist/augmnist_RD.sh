#!/usr/bin/env bash

experiment="augmnist_RD"
notes="
**Goal**: RD curves for mnist that is augmented at test and training time (i.e. when we assume we know the augmentations),
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
is_only_feat=False
featurizer=neural_rec
architecture@encoder=resnet18
architecture@predictor=resnet18
data@data_feat=mnist_aug
rate=H_hyper
trainer.max_epochs=100
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=VIC,VAE
featurizer.loss.beta=0.0001,0.001,0.01,0.03,0.1,0.3,1,3,10,100
seed=1,2,3
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi


wait



#for featurizer
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_err','test/pred/loss','test/pred/err']"
compare="dist"
data="merged" # want to access both ther featurizer data and the  predictor data
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       +summarize_RD_curves.data="${data}" \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="${distortion_cols}" \
       +plot_all_RD_curves.data="${data}" \
       +plot_all_RD_curves.rate_cols="${rate_cols}" \
       +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
       +plot_all_RD_curves.hue=$compare \
       +plot_invariance_RD_curve.data="${data}" \
       +plot_invariance_RD_curve.noninvariant='VAE' \
       +plot_invariance_RD_curve.logbase_x=2 \
       +plot_invariance_RD_curve.desirable_distortion="test/pred/loss" \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_invariance_RD_curve] || true #  make sure continue even if error


#plot loaded model
col_val_subset=""
python utils/load_pretrained.py \
      load_pretrained.experiment=$experiment  \
      $col_val_subset \
      $kwargs  \
      server=none \
      trainer.gpus=0 \
      $kwargs_multi \
      load_pretrained.mode=[latent_traversals_plot,reconstruct_image_plot] \
      -m 

col_val_subset=""
python utils/load_pretrained.py \
      load_pretrained.experiment=$experiment  \
      $col_val_subset \
      $kwargs  \
      server=none \
      trainer.gpus=0 \
      distortion=VIC \
      featurizer.loss.beta=0.1 \
      seed=1 \
      +load_pretrained.reconstruct_image_plot_placeholder.is_single_row=True \
      +load_pretrained.reconstruct_image_plot_placeholder.add_standard='\ \(130 Bits\)' \
      +load_pretrained.reconstruct_image_plot_placeholder.add_invariant='\ \(48 Bits\)' \
      load_pretrained.mode=[reconstruct_image_plot_placeholder] \
      -m

col_val_subset=""
python utils/load_pretrained.py \
      load_pretrained.experiment=$experiment  \
      $col_val_subset \
      $kwargs  \
      server=none \
      trainer.gpus=0 \
      distortion=VIC \
      featurizer.loss.beta=0.1 \
      seed=1 \
      +load_pretrained.reconstruct_image_plot_placeholder.is_single_row=False \
      +load_pretrained.reconstruct_image_plot_placeholder.add_standard='\ \(130 Bits\)' \
      +load_pretrained.reconstruct_image_plot_placeholder.add_invariant='\ \(48 Bits\)' \
      load_pretrained.mode=[reconstruct_image_plot_placeholder] \
      -m
