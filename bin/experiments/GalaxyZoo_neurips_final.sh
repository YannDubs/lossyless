#!/usr/bin/env bash


export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="GalaxyZoo_neurips_table_final"
notes="
**Goal**: GalaxyZoo experiments from Neurips paper, jpeg,webp,VAE,INCE comparison, predictions as appropriate ca 100 runs each
"
# parses special mode for running the script
source `dirname $0`/../utils.sh

# project and server kwargs
kwargs="
logger.kwargs.project=lossyless
wandb_entity=${env:USER}
experiment=$experiment
timeout=$time
$add_kwargs
"

# experiment kwargs
kwargs="
$kwargs
is_only_feat=False
data@data_feat=galaxy256
rate=H_hyper
checkpoint@checkpoint_pred=bestValLoss
checkpoint@checkpoint_feat=bestTrainLoss
trainer.max_epochs=100
+update_trainer_pred.max_epochs=100
"

# classical sweeping arguments
kwargs_hypopt_jpeg="
featurizer=jpeg++,webp++
evaluation.featurizer.is_evaluate=False
featurizer.quality=1,2,3,5,10,20,50,70,95
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=200
hydra.sweeper.n_jobs=200
monitor_direction=[minimize]
monitor_return=[val/pred/loss]
seed=0,1,2,3,4,5,6,7,8,9
+data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam
architecture@predictor=resnet50
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=plateau_quick,cosine_restart,expdecay100,unifmultistep100
"

# VAE sweeping arguments
kwargs_hypopt_vae_balle="
distortion=vae
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=100
monitor_direction=[minimize,minimize]
monitor_return=[val/feat/rate,val/pred/loss]
featurizer=neural_rec
rate=H_spatial
architecture@encoder=balle 
+distortion.kwargs.arch=balle
encoder.z_dim=65536,32768,16384,8192,4096
evaluation.featurizer.is_online=false
data_feat.kwargs.batch_size=tag(log,int(interval(32,64)))
featurizer.loss.beta=tag(log,interval(1e-12,1e-4))
distortion.factor_beta=tag(log,interval(1e-5,1))
rate.kwargs.warmup_k_epoch=int(interval(0,3))
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-5))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,1e-3))
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-5))
optimizer_coder.kwargs.lr=tag(log,interval(3e-4,1e-3))
scheduler@scheduler_feat=expdecay100,unifmultistep100,unifmultistep1000
scheduler@scheduler_coder=expdecay100,unifmultistep100
seed=0,1,2,3,4,5,6,7,8,9
architecture@predictor=resnet50
+data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=plateau_quick,cosine_restart,expdecay100,unifmultistep100
"

# VAE sweeping arguments
kwargs_hypopt_vae="
distortion=vae
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=100
monitor_direction=[minimize,minimize]
monitor_return=[val/feat/rate,val/pred/loss]
featurizer=neural_rec
architecture@encoder=resnet50 
data_feat.kwargs.batch_size=tag(log,int(interval(32,64)))
encoder.z_dim=512,2048
featurizer.loss.beta=tag(log,interval(1e-12,1e-4))
distortion.factor_beta=tag(log,interval(1e-5,1))
rate.kwargs.warmup_k_epoch=int(interval(0,3))
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-5))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,1e-3))
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-5))
optimizer_coder.kwargs.lr=tag(log,interval(3e-4,1e-3))
scheduler@scheduler_feat=expdecay100,unifmultistep100,unifmultistep1000
scheduler@scheduler_coder=expdecay100,unifmultistep100
seed=0,1,2,3,4,5,6,7,8,9
architecture@predictor=resnet50
+data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=plateau_quick,cosine_restart,expdecay100,unifmultistep100
"

# ince sweeping arguments
kwargs_hypopt_ince="
distortion=ince,ince_basic
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=200
hydra.sweeper.n_jobs=200
monitor_direction=[minimize,minimize]
monitor_return=[val/feat/rate,val/pred/loss]
featurizer=neural_feat
architecture@encoder=resnet50
data_feat.kwargs.batch_size=tag(log,int(interval(64,128)))
encoder.z_dim=512,2048
featurizer.loss.beta=tag(log,interval(1e-12,1e-4))
distortion.factor_beta=tag(log,interval(1e-5,1))
distortion.kwargs.is_train_temperature=true,false
distortion.kwargs.temperature=0.1,0.01
distortion.kwargs.project_kwargs.out_shape=128,256
seed=0,1,2,3,4,5,6,7,8,9
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-5))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,1e-3))
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-5))
optimizer_coder.kwargs.lr=tag(log,interval(3e-4,1e-3))
scheduler@scheduler_feat=expdecay100,unifmultistep100,unifmultistep1000
scheduler@scheduler_coder=expdecay100,unifmultistep100
architecture@predictor=mlp_probe
+data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=plateau_quick,cosine_restart,expdecay100,unifmultistep100
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
featurizer.is_on_the_fly=false,true
"

# ince sweeping arguments
kwargs_hypopt_clip="
featurizer=bottleneck_clip_lossyZ
data_feat.kwargs.dataset_kwargs.equivalence=[resize_crop,D4_group,color,gray]
data_pred.kwargs.dataset_kwargs.equivalence=[resize_crop,D4_group,color,gray]
data_feat.kwargs.dataset_kwargs.is_normalize=true
data_pred.kwargs.dataset_kwargs.is_normalize=true
featurizer.loss.beta=tag(log,interval(1e-4,1e-1))
distortion.factor_beta=tag(log,interval(1e-4,1e-1))
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=100
monitor_direction=[minimize,minimize]
monitor_return=[val/feat/rate,val/pred/loss]
encoder.z_dim=512
seed=0,1,2,3,4,5,6,7,8,9
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,1e-6))
optimizer_feat.kwargs.lr=tag(log,interval(1e-5,1e-4))
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,1e-5))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_feat=expdecay100,expdecay1000
scheduler@scheduler_coder=expdecay100,expdecay1000
architecture@predictor=mlp_probe
+data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-7,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=plateau_quick,cosine_restart,expdecay100,unifmultistep100
predictor.arch_kwargs.dropout_p=interval(0.,0.5)
featurizer.is_on_the_fly=false,true
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_ince $kwargs_dep -m &

    sleep 2

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_clip $kwargs_dep -m &
    
    sleep 2

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_ivae $kwargs_dep -m &

    sleep 2

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_vae_balle $kwargs_dep -m &

    sleep 2

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_jpeg $kwargs_dep -m &
    
  done
fi

