#!/usr/bin/env bash

experiment="dev_augmnist_ivae"
notes="
**Goal**: RD curves for mnist
"

# e.g. command: bin/experiments/cifar10_RD_compare_rates.sh -s vector -t 1440

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
encoder.z_dim=128
data@data_feat=augmnist
evaluation.is_est_entropies=True
rate=H_hyper
optimizer@optimizer_pred=sgd 
scheduler@scheduler_pred=multistep
trainer.max_epochs=100
scheduler_pred.kwargs.MultiStepLR.milestones=[20,40,60,80]
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=ivae
featurizer.loss.beta=0.1
seed=1
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
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_acc','test/pred/loss','test/pred/acc']"
compare="dist"
data="merged" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
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
       +plot_invariance_RD_curve.noninvariant='vae' \
       +plot_invariance_RD_curve.logbase_x=2 \
       +plot_invariance_RD_curve.desirable_distortion="test/pred/loss" \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_invariance_RD_curve] || true #  make sure continue even if error

# col_val_subset=""
# rate_cols="['test/feat/rate']"
# distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_acc']"
# compare="dist"
# data="featurizer" # want to access both ther featurizer data and the  predictor data
# python aggregate.py \
#        experiment=$experiment  \
#        $col_val_subset \
#        collect_data.predictor=null \
#        +summarize_RD_curves.data="${data}" \
#        +summarize_RD_curves.rate_cols="${rate_cols}" \
#        +summarize_RD_curves.distortion_cols="${distortion_cols}" \
#        +summarize_RD_curves.mse_cols="${distortion_cols}" \
#        +plot_all_RD_curves.data="${data}" \
#        +plot_all_RD_curves.rate_cols="${rate_cols}" \
#        +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
#        +plot_all_RD_curves.hue=$compare \
#        agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves]



# plot loaded model
col_val_subset=""
python load_pretrained.py \
      load_pretrained.experiment=$experiment  \
      $col_val_subset \
      $kwargs  \
      server=none \
      trainer.gpus=0 \
      $kwargs_multi \
      load_pretrained.mode=[latent_traversals_plot,reconstruct_image_plot] \
      -m 