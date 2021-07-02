#!/usr/bin/env bash

experiment="hyp_clip_eval_stl10"
notes="
**Goal**: Test and tune hyperaparmeters for staggered clip
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
data@data_pred=stl10
checkpoint@checkpoint_feat=bestValLoss
trainer.max_epochs=50
+update_trainer_pred.max_epochs=150
featurizer=bottleneck_clip_lossyZ
data_feat.kwargs.num_workers=4
featurizer.is_on_the_fly=false
optimizer@optimizer_pred=Adam
featurizer.loss.beta_anneal=linear
paths.pretrained.load=/scratch/ssd002/home/yannd/projects/lossyless/pretrained/exp_hyp_clip_lossyZ_cifar10/datafeat_coco/feat_bottleneck_clip_lossyZ/dist_lossyZ/enc_clip/rate_H_hyper/optfeat_Adam_lr7.9e-05_w2.2e-06/schedfeat_plateau_quick/zdim_512/zs_1/beta_3.4e-02/seed_4/addfeat_None/jid_2_2727892_2
featurizer.is_train=false
evaluation.communication.ckpt_path=null
$add_kwargs
"

# sweeping arguments
kwargs_hypopt="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=50
monitor_direction=[minimize]
monitor_return=[test/pred/err]
"

# FEATURIZER
# if the values that are swept over are not understandable from the names `interval` `log`.. check : https://hydra.cc/docs/next/plugins/optuna_sweeper
kwargs_multi="
$kwargs_hypopt
seed=0,1,2,3,4,5,6,7,8,9
data_pred.kwargs.batch_size=tag(log,int(interval(16,64)))
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-6,5e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-5,3e-3))
scheduler@scheduler_pred=cosine_restart,expdecay100,plateau_quick,unifmultistep1000
predictor.arch_kwargs.dropout_p=interval(0.2,0.5)
" 


kwargs_multi="
trainer.max_epochs=1
++update_trainer_pred.max_epochs=5
mode=dev
+update_trainer_pred.limit_val_batches=1.0
+update_trainer_pred.limit_test_batches=0.5
data_pred.kwargs.batch_size=64
optimizer_pred.kwargs.lr=0.1
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""         
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep 

    sleep 7

    
  done
fi
