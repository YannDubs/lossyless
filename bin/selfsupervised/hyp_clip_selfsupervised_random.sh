#!/usr/bin/env bash

experiment=$prfx"hyp_clip_selfsupervised_random"
notes=""

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
dataset=ImagenetDataset
model=CLIP_ViT
logger.kwargs.project=selfsupervised
trainer.max_epochs=20
architecture=fancymlp
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_hypopt="
hydra/sweeper=optuna
hypopt=optuna
hydra.sweeper.optuna_config.n_trials=225
hydra.sweeper.optuna_config.n_jobs=75
hydra.sweeper.optuna_config.sampler=random
"

kwargs_multi="
$kwargs_hypopt
architecture.arch_kwargs.hid_dim=tag(log,int(interval(512,4096)))
architecture.arch_kwargs.dropout_p=interval(0.,0.5)
architecture.arch_kwargs.n_hid_layers=1,2
data_kwargs.batch_size=tag(log,int(interval(32,128)))
trainer.gradient_clip_val=tag(log,interval(0.3,10))
optimizer=SGD_likeadam,Adam,AdamW
optimizer.kwargs.weight_decay=tag(log,interval(1e-6,5e-4))
optimizer.kwargs.lr=tag(log,interval(1e-4,5e-3))
scheduler=cosine,cosine_restart,expdecay1000,expdecay100,plateau_vquick,unifmultistep100
seed=0,1,2,3,4
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" 
  do

    python bin/selfsupervised/selfsupervised.py $kwargs $kwargs_multi $kwargs_dep  -m & 
    
  done
fi

#for featurizer
col_val_subset=""
distortion_cols="['test/pred/loss','test/pred/err']"
data="predictor" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       collect_data.featurizer=null \
       agg_mode=[plot_optuna_hypopt] 
