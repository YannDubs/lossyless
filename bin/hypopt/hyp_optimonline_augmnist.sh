#!/usr/bin/env bash

experiment="hyp_optimonline_augmnist"
notes="
**Goal**: Tuning learning rate / scheduler / optimizer of the online for ivae on augmented mnist
"


# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
is_only_feat=True
featurizer=neural_rec
architecture@encoder=resnet18
encoder.z_dim=128
data@data_feat=augmnist
evaluation.is_est_entropies=False
rate=H_factorized
trainer.max_epochs=100
data_feat.kwargs.batch_size=128
distortion=ivae
featurizer.loss.beta=0.1
seed=1
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
optimizer@optimizer_coder=adam1e-3,adam3e-3,adam1e-4,sgd,sgd05,sgd005
scheduler@scheduler_coder=multistep100,cosine,expdecay,plateau,none
" 


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7
    
  done
fi


wait

#for featurizer
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_acc']"
data="featurizer" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       +summarize_RD_curves.data="${data}" \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="${distortion_cols}" \
       agg_mode=[summarize_metrics,summarize_RD_curves] 
