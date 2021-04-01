#!/usr/bin/env bash

experiment=$prfx"hyp_clip_sslfeat_cifar10"
notes=""

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
dataset=CIFAR10
model=CLIP_ViT
logger.kwargs.project=selfsupervised
trainer.max_epochs=20
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_hypopt="
hydra/sweeper=optuna
hypopt=optuna
hydra.sweeper.optuna_config.n_trials=150
hydra.sweeper.optuna_config.n_jobs=50
hydra.sweeper.optuna_config.sampler=random
"

kwargs_multi="
$kwargs_hypopt
featurizer.loss.beta=tag(log,interval(1e-8,1e2))
featurizer.loss.beta_anneal=linear,constant
online_evaluator.arch_kwargs.hid_dim=tag(log,int(interval(512,4096)))
online_evaluator.arch_kwargs.dropout_p=interval(0.,0.5)
online_evaluator.arch_kwargs.n_hid_layers=1,2
optimizer@optimizer_feat=Adam,AdamW,SGD_likeadam
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,1e-2))
optimizer@optimizer_online=SGD_likeadam,Adam,AdamW
optimizer_online.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_online.kwargs.lr=tag(log,interval(1e-4,1e-2))
optimizer@optimizer_coder=SGD_likeadam,Adam,AdamW
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,3e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,plateau,unifmultistep1000
scheduler@scheduler_online=cosine_restart,expdecay100,expdecay1000,plateau_quick,unifmultistep100
scheduler@scheduler_coder=cosine_restart,expdecay100,plateau_quick,unifmultistep1000,unifmultistep100
data.kwargs.batch_size=tag(log,int(interval(32,128)))
trainer.gradient_clip_val=tag(log,interval(0.3,10))
seed=0,1,2,3,4
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" 
  do

    python bin/selfsupervised/selfsupervised_feat.py $kwargs $kwargs_multi $kwargs_dep  -m & 
    
  done
fi

wait

#for featurizer
col_val_subset=""
distortion_cols="['test/feat/loss','test/feat/err']"
data="featurizer" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       collect_data.featurizer=null \
       agg_mode=[plot_optuna_hypopt] 
