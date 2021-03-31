#!/usr/bin/env bash

export MKL_SERVICE_FORCE_INTEL=1 # avoid learnfair error
export HYDRA_FULL_ERROR=1

experiment="hyp_cifar10_distortion_variation_feat"
notes="
**Goal**: Test different distortions
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

kwargs="
experiment=$experiment
timeout=$time
is_only_feat=False LOAD PRETRAINED
featurizer=neural_feat
architecture@predictor=resnet18
encoder.z_dim=128
data@data_feat=cifar10
evaluation.is_est_entropies=True
optimizer@optimizer_pred=sgd
scheduler@scheduler_pred=multistep
scheduler_pred.kwargs.MultiStepLR.milestones=[60,120,160]
trainer.max_epochs=200
+data.kwargs.dataset_kwargs.equivalence=[auto_cifar10]
$add_kwargs
rate=H_hyper
distortion.kwargs.arch_kwargs.complexity=3
optimizer_pred.kwargs.weight_decay=0.001
"

# every arguments that you are sweeping over
kwargs_multi="
seed=1
featurizer.loss.beta=0.1
distortion=ivae,vae
optimizer_feat.kwargs.lr=0.0001,0.0005,0.001
architecture@predictor=fancymlp,mlp1024
scheduler_pred.kwargs.MultiStepLR.gamma=0.15
optimizer_pred.kwargs.lr=0.05,0.08,0.1
"
# "distortion=ince"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do
    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    sleep 3
  done
fi

wait

# for featurizer
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/pred/loss','test/pred/acc']"
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
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves]



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