#!/usr/bin/env bash

experiment="hypopt_clip_bottleneck"
notes="
**Goal**: Hyperparameter tunning for CLIP with an additional entropy bottleneck
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
logger.kwargs.project=neurips
experiment=$experiment 
timeout=$time
encoder.z_dim=512
is_only_feat=False
data@data_feat=coco
data@data_pred=stl10
checkpoint@checkpoint_feat=bestValLoss
checkpoint@checkpoint_pred=bestValLoss
trainer.max_epochs=50
+update_trainer_pred.max_epochs=100
featurizer=bottleneck_clip_lossyZ
data_feat.kwargs.num_workers=4
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=200
hydra.sweeper.n_jobs=50
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=64
featurizer.loss.beta_anneal=linear
featurizer.loss.beta=tag(log,interval(3e-3,5e-2))
distortion.factor_beta=tag(log,interval(3e-3,5e-2))
optimizer@optimizer_feat=AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-6))
optimizer_feat.kwargs.lr=tag(log,interval(3e-5,3e-4))
optimizer@optimizer_coder=Adam,AdamW
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-5))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,5e-4))
scheduler@scheduler_feat=expdecay1000,expdecay100,unifmultistep100,unifmultistep1000,cosine_restart,plateau_quick
scheduler@scheduler_coder=expdecay1000,expdecay100
rate.kwargs.is_endToEnd=False
distortion.p_norm=1,2
seed=0,1,2,3,4,5,6,7,8,9
" 
# seeds are not tuned over but swept because using random sampler

# PREDICTOR (fixed for simpler comparison)
kwargs_multi="
$kwargs_multi
architecture@predictor=mlp_probe
data_pred.kwargs.batch_size=64
optimizer_pred.kwargs.weight_decay=1e-5
optimizer_pred.kwargs.lr=3e-4
scheduler@scheduler_pred=plateau_quick
predictor.arch_kwargs.dropout_p=0.3
optimizer@optimizer_pred=Adam
featurizer.is_on_the_fly=false
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi


wait


col_val_subset=""
rate_cols="['test/comm/rate']"
distortion_cols="['test/feat/distortion','test/pred/err','test/pred/loss']"
compare="dist"
data="merged" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       +summarize_RD_curves.data="${data}" \
       +summarize_RD_curves.rate_cols="${rate_cols}" \
       +summarize_RD_curves.distortion_cols="${distortion_cols}" \
       +summarize_RD_curves.mse_cols="[]" \
       +plot_all_RD_curves.data="${data}" \
       +plot_all_RD_curves.rate_cols="${rate_cols}" \
       +plot_all_RD_curves.distortion_cols="${distortion_cols}" \
       +plot_all_RD_curves.hue=$compare \
       agg_mode=[summarize_metrics,summarize_RD_curves,plot_all_RD_curves,plot_optuna_hypopt] 