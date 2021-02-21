#!/usr/bin/env bash

experiment="hyp_cifar10_featrec"
notes="
**Goal**: RD curves for cifar10
"

# e.g. command: bin/experiments/cifar10_RD_compare_rates.sh -s vector -t 1440

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
is_only_feat=False
architecture@encoder=resnet18
distortion.kwargs.arch_kwargs.complexity=3
encoder.z_dim=128
data@data_feat=cifar10
evaluation.is_est_entropies=True
rate=H_hyper
optimizer@optimizer_pred=sgd 
scheduler@scheduler_pred=multistep
trainer.max_epochs=200
+data.kwargs.dataset_kwargs.equivalence=[auto_cifar10]
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=ivae
seed=1
" 


# seed=1,2,3,4,5

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "featurizer.loss.beta=0.1 architecture@predictor=resnet18 featurizer=neural_rec" "featurizer.loss.beta=0.1 architecture@predictor=fancymlp featurizer=neural_feat"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi


wait

# for featurizer
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/pred/loss']"
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
       +plot_all_RD_curves.logbase_x=2 \
       +plot_all_RD_curves.hue=$compare \
       +plot_invariance_RD_curve.data="${data}" \
       +plot_invariance_RD_curve.noninvariant='vae' \
       +plot_invariance_RD_curve.logbase_x=2 \
       +plot_invariance_RD_curve.desirable_distortion="test/pred/loss" \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_invariance_RD_curve]



# plot loaded model
col_val_subset=""
python load_pretrained.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       server=local \
       trainer.gpus=0 \
       $kwargs_multi \
       $kwargs_multi \
       load_pretrained.mode=[latent_traversals_plot,reconstruct_image_plot] \
       -m