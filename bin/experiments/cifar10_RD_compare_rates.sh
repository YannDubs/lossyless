#!/usr/bin/env bash

experiment="cifar10_RD_compare_rates"
notes="
**Goal**: comparing the variational bounds of rates
**Hypothesis**: hyper and vamp should be better than factorized and gaussian respectively
**Results**: table with AURD and high fidellity (lossless prediction) rate, and rate-loss curve
"

# e.g. command: bin/experiments/cifar10_RD_compare_rates.sh -s vector -t 1440

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
is_only_feat=False
featurizer=neural_feat
architecture@encoder=resnet18
architecture@predictor=fancymlp
data@data_feat=cifar10
evaluation.is_est_entropies=True
distortion=ivae
trainer.max_epochs=200
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
rate=H_factorized,H_hyper,MI_vamp,MI_unitgaussian
featurizer.loss.beta=0.001,0.01,0.03,0.1,0.3,1,3,10,100
seed=1
" 
# seed=1,2,3,4,5

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
distortion_cols="['test/feat/distortion','test/feat/online_loss']"
compare="dist"
python aggregate.py \
       experiment=$experiment  \
       collect_data.predictor=null \
       $col_val_subset \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="${distortion_cols}" \
       +plot_all_RD_curves.rate_cols="${rate_cols}" \
       +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
       +plot_all_RD_curves.logbase_x=null \
       +plot_all_RD_curves.hue=$compare \
       +plot_invariance_RD_curve.noninvariant='vae' \
       +plot_invariance_RD_curve.logbase_x=null \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_invariance_RD_curve]


#TODO plotting for predictor