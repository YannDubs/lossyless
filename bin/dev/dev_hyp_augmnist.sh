#!/usr/bin/env bash

experiment="dev_hyp_augmnist"
notes="
**Goal**: Hyperparameter tuning for ince on augmented mnist
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
evaluation.is_est_entropies=False
rate=H_factorized
trainer.max_epochs=50
distortion=ivae
seed=1
$add_kwargs
"

kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
is_only_feat=True
featurizer=neural_feat
architecture@encoder=resnet18
data@data_feat=augmnist
evaluation.is_est_entropies=False
rate=H_factorized
trainer.max_epochs=2
+trainer.limit_val_batches=0.2
+trainer.limit_train_batches=0.01
+trainer.limit_test_batches=0.1
distortion=ince
seed=1
$add_kwargs
"

kwargs_hypopt="
hydra/sweeper=optuna
hypopt=multi_optuna
hydra.sweeper.optuna_config.n_trials=4
hydra.sweeper.optuna_config.n_jobs=2
monitor_direction=[minimize,minimize]
monitor_return=[test/feat/online_err,test/feat/rate]
"

kwargs_multi="
$kwargs_hypopt
data_feat.kwargs.batch_size=tag(log,int(interval(32,256)))
encoder.z_dim=tag(log,int(interval(8,4096)))
featurizer.loss.beta=tag(log,interval(1e-6,1e2))
featurizer.loss.beta_anneal=linear,constant,geometric
rate.kwargs.warmup_k_epoch=int(interval(0,5))
online_evaluator.arch_kwargs.hid_dim=tag(log,int(interval(512,4096)))
online_evaluator.arch_kwargs.norm_layer=batchnorm,identity
online_evaluator.arch_kwargs.dropout_p=interval(0.,0.5)
online_evaluator.arch_kwargs.n_hid_layers=1,2
online_evaluator.arch_kwargs.activation=ReLU,Softplus
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-3))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,1e-2))
optimizer_feat.kwargs.is_lars=true,false
optimizer@optimizer_online=SGD_likeadam,Adam,AdamW
optimizer_online.kwargs.weight_decay=tag(log,interval(1e-8,1e-3))
optimizer_online.kwargs.lr=tag(log,interval(1e-4,1e-2))
optimizer@optimizer_coder=SGD_likeadam,Adam,AdamW
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-8,1e-3))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,1e-2))
scheduler@scheduler_feat=cosine,cosine_restart,expdecay100,expdecay1000,plateau_quick,plateau,unifmultistep1000,unifmultistep100,none
scheduler@scheduler_online=cosine,cosine_restart,expdecay100,expdecay1000,plateau_quick,plateau,unifmultistep1000,unifmultistep100,none
scheduler@scheduler_coder=cosine,cosine_restart,expdecay100,expdecay1000,plateau_quick,plateau,unifmultistep1000,unifmultistep100,none
" 
# NB `SGD_likeadam` mulitiplies lr of sgd by 100 so that lr is similar to Adam (easier for sweeping)



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
distortion_cols="['test/feat/distortion','test/feat/online_loss','test/feat/online_err']"
data="featurizer" # want to access both ther featurizer data and the  predictor data
python aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       collect_data.predictor=null \
       agg_mode=[summarize_metrics,plot_optuna_hypopt] 
