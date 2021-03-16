#!/usr/bin/env bash

export MKL_SERVICE_FORCE_INTEL=1 # avoid learnfair error

experiment="galaxy_RD_workshop"
notes=" "

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
data@data_feat=galaxy64
architecture@encoder=resnet18
architecture@predictor=resnet18
distortion.kwargs.arch_kwargs.complexity=3
encoder.z_dim=128
rate=H_hyper
is_only_feat=False
optimizer@optimizer_pred=sgd 
scheduler@scheduler_pred=multistep
evaluation.is_est_entropies=True
trainer.max_epochs=200
$add_kwargs
logger=tensorboard
"

# every arguments that you are sweeping over
# see kwargs_dep for conditional sweeping
kwargs_multi="
seed=1,2,3
"


if [ "$is_plot_only" = false ] ; then
  # this performs sweepingfor dependent / conditional arguments
  for kwargs_dep in  "featurizer=neural_rec distortion=vae,ivae featurizer.loss.beta=0.00001,0.0001,0.001,0.01,0.03,0.1,0.3,1"  "featurizer=jpeg++ featurizer.quality=10,20,30,40,50,60,70,80,90" "featurizer=none"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi

wait #only plot when finished running

# for featurizer
# this will most probably give an error because it hasn't been tested on jpeg and classical compressors, but we can deal with it later
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/pred/loss']" #TODO add everything you are logging and can be seen as a distortion
compare="dist"
data="merged" # want to access both the featurizer data and the predictor data
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
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves]



# plot loaded model
# this will load each model that you saved and do some plotting (non aggregated)
col_val_subset=""
python load_pretrained.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       server=local \
       trainer.gpus=0 \
       $kwargs_multi \
       load_pretrained.mode=[latent_traversals_plot,reconstruct_image_plot] \
       -m