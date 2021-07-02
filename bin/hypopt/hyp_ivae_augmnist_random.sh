#!/usr/bin/env bash

experiment="hyp_ivae_augmnist_random"
notes="
**Goal**: Hyperparameter tuning for ivae on augmented mnist
"


# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
is_only_feat=True
featurizer=neural_feat
architecture@encoder=resnet18
data@data_feat=augmnist
rate=H_factorized
trainer.max_epochs=50
distortion=ivae
seed=1
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hypopt=multi_optuna
hydra.sweeper.optuna_config.n_trials=225
hydra.sweeper.optuna_config.n_jobs=75
monitor_direction=[minimize,minimize]
monitor_return=[test/feat/online_err,test/feat/rate]
hydra.sweeper.optuna_config.sampler=random
"

kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=tag(log,int(interval(32,64)))
encoder.z_dim=tag(log,int(interval(16,512)))
featurizer.loss.beta=tag(log,interval(1e-6,1e2))
featurizer.loss.beta_anneal=linear,constant,geometric
rate.kwargs.warmup_k_epoch=int(interval(0,5))
online_evaluator.arch_kwargs.hid_dim=tag(log,int(interval(512,4096)))
online_evaluator.arch_kwargs.dropout_p=interval(0.,0.3)
online_evaluator.arch_kwargs.n_hid_layers=1,2
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer@optimizer_online=SGD_likeadam,Adam,AdamW
optimizer_online.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_online.kwargs.lr=tag(log,interval(1e-4,1e-3))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,3e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau,unifmultistep1000
scheduler@scheduler_online=cosine,cosine_restart,expdecay100,expdecay1000,unifmultistep100
scheduler@scheduler_coder=cosine_restart,expdecay100,plateau_quick,unifmultistep100
" 
# n_lay = 2 is better


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""   
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7
    
  done
fi


wait




#for featurizer
col_val_subset=""
rate_cols="['test/feat/rate']"
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_err']"
data="featurizer" # want to access both ther featurizer data and the  predictor data
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       collect_data.predictor=null \
       agg_mode=[summarize_metrics,plot_optuna_hypopt] 