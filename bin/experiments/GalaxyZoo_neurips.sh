#!/usr/bin/env bash


export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="GalaxyZoo_neurips_table"
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
data@data_feat=galaxy64
rate=H_hyper
trainer.max_epochs=100
+update_trainer_pred.max_epochs=100
"

# classical sweeping arguments
kwargs_hypopt_jpeg="
featurizer=jpeg++,png,webp++
evaluation.featurizer.is_evaluate=False
featurizer.quality=1,3,5,10,20,30,40,70,95
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=300
hydra.sweeper.n_jobs=300
monitor_direction=[minimize]
monitor_return=[val/pred/loss]
seed=0,1,2,3,4,5,6,7,8,9
+data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,unifmultistep100
"

# VAE sweeping arguments
kwargs_hypopt_ivae="
distortion=vae,ivae
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=200
hydra.sweeper.n_jobs=200
monitor_direction=[minimize,minimize]
monitor_return=[val/feat/rate,val/pred/loss]
featurizer=neural_rec
architecture@encoder=resnet18 #b=0, balle, cnn?, hidden dim with z_dim
data_feat.kwargs.batch_size=tag(log,int(interval(32,128)))
encoder.z_dim=tag(log,int(interval(128,512))) # 4000
featurizer.loss.beta_anneal=linear
featurizer.loss.beta=tag(log,interval(1e-8,1e0))
distortion.factor_beta=tag(log,interval(1e-5,100))
rate.kwargs.warmup_k_epoch=int(interval(0,3))
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,plateau
scheduler@scheduler_coder=cosine_restart,expdecay100,plateau_quick,unifmultistep100
seed=0,1,2,3,4,5,6,7,8,9
+data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,expdecay1000,unifmultistep100
"


# ince sweeping arguments
kwargs_hypopt_ince="
distortion=ince
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
hydra.sweeper.n_trials=100
hydra.sweeper.n_jobs=100
monitor_direction=[minimize,minimize]
monitor_return=[val/feat/rate,val/pred/loss]
featurizer=neural_feat
architecture@encoder=resnet18
data_feat.kwargs.batch_size=128
encoder.z_dim=tag(log,int(interval(32,512)))
eaturizer.loss.beta=tag(log,interval(1e-5,1e-1))
distortion.factor_beta=tag(log,interval(1e-5,1e-1))f
seed=0,1,2,3,4,5,6,7,8,9
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=1e-5
optimizer_feat.kwargs.lr=3e-4
optimizer@optimizer_coder=Adam
optimizer_coder.kwargs.weight_decay=1e-6
optimizer_coder.kwargs.lr=1e-4
scheduler@scheduler_feat=expdecay1000,expdecay100,unifmultistep100,unifmultistep1000
scheduler@scheduler_coder=expdecay100
rate.kwargs.invertible_processing=diag
distortion.kwargs.is_train_temperature=false
architecture@predictor=mlp_probe
+data_pred.kwargs.batch_size=64
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

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_ince $kwargs_dep -m &

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_ivae $kwargs_dep -m &

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_jpeg $kwargs_dep -m &

    sleep 7
  done
fi

