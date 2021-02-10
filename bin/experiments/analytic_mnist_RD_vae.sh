#!/usr/bin/env bash

experiment="analytic_mnist_RD_vae"
notes="
**Goal**: Understand how the gains changing at different points of the rate distortion curve for VAE and iVAE when looking at H[M(X)|Z] and the upperbound H[X|Z], Showing that the empirical resutls follow the schematic RD curve that should happen in theory.
**Hypothesis**: should follow closely schematic polot. So gains compared to upper bound should be constant but actual gains can fall in between  
**Plot**: a RD curve for iVAE with invariance distortion, VAE with invariance distortion, and VAE with non invariance distoriton (i.e. upper bound), and write down theoretical gains 
"

#! UPDATE: this is not analytic actually, all we can say is the maximal gains that you can have not the actual gains that you can have.

# e.g. command: bin/experiments/analytic_mnist_RD_vae.sh -s vector -t 720

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
is_only_feat=True
featurizer=neural_rec
architecture@encoder=resnet18
rate=H_factorized
data@data_feat=analytic_mnist
evaluation.is_est_entropies=True
trainer.max_epochs=200
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
distortion=ivae,vae
featurizer.loss.beta=0.001,0.01,0.03,0.1,0.3,1,3,10,100
seed=1,2,3
" 
#seed=1,2,3,4,5

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
