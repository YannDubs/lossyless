#!/usr/bin/env bash

experiment="hyp_entropybottleneck_augmnist"
notes="
**Goal**: Tuning of entropy bottleneck parameters on augmented mnist
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
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "rate.kwargs.kwargs_ent_bottleneck.filters=[3,3,3],[3,3],[3,3,3,3,3]"  #"rate.kwargs.kwargs_ent_bottleneck.init_scale=5,10,50"
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
