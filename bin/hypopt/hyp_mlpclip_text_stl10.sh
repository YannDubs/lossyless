#!/usr/bin/env bash

experiment="hyp_mlpclip_text_stl10"
notes="
**Goal**: Test whether adds better by adding MLP on CLIP
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
logger.kwargs.project=hypopt
encoder.z_dim=512
is_only_feat=False
architecture@predictor=mlp_probe
data@data_feat=coco
data@data_pred=cifar100
checkpoint@checkpoint_feat=bestValLoss
featurizer.is_on_the_fly=false
trainer.max_epochs=30
+update_trainer_pred.max_epochs=150
featurizer=bottleneck_mlpclip
data_feat.kwargs.num_workers=4
rate.kwargs.invertible_processing=diag
data_feat.kwargs.batch_size=128
rate.kwargs.warmup_k_epoch=1
distortion.kwargs.is_train_temperature=true
distortion.kwargs.temperature=1e-2
optimizer@optimizer_pred=Adam
optimizer@optimizer_feat=Adam
featurizer.loss.beta_anneal=linear
$add_kwargs
"

# data_pred should not be augmented (it won't here due to no ot the fly but better if it was explicit)
# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=50
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
"

kwargs_multi="
$kwargs_hypopt
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
featurizer.loss.beta=tag(log,interval(1e-6,1e-3))
distortion.factor_beta=tag(log,interval(1e-4,1e-1))
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-5,1e-4))
optimizer_feat.kwargs.lr=tag(log,interval(5e-6,5e-4))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-5,1e-3))
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-6,5e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,3e-3))
scheduler@scheduler_feat=expdecay100,plateau_quick
scheduler@scheduler_coder=expdecay1000,unifmultistep1000,cosine,plateau_quick
scheduler@scheduler_pred=cosine_restart,cosine,expdecay1000,plateau_quick,unifmultistep1000
seed=0,1,2,3,4,5,6,7,8,9
predictor.arch_kwargs.dropout_p=interval(0.1,0.4)
" 



# kwargs_multi="
# trainer.max_epochs=1
# ++update_trainer_pred.max_epochs=1
# mode=dev
# +update_trainer_pred.limit_val_batches=1.0
# +update_trainer_pred.limit_test_batches=0.5
# evaluation.featurizer.is_evaluate=True
# data_feat.kwargs.batch_size=64
# data_pred.kwargs.batch_size=64
# optimizer_pred.kwargs.lr=1e-3
# featurizer.loss.beta=1e-3
# distortion.factor_beta=1e-3
# seed=0
# optimizer@optimizer_feat=Adam
# optimizer_feat.kwargs.weight_decay=5e-5
# optimizer_feat.kwargs.lr=1e-4
# optimizer_coder.kwargs.weight_decay=5e-4
# optimizer_coder.kwargs.lr=5e-4
# scheduler@scheduler_feat=expdecay100
# scheduler@scheduler_coder=expdecay100
# "

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 7

    
  done
fi
