#!/usr/bin/env bash


export MKL_SERVICE_FORCE_INTEL=1 # avoid server error
export HYDRA_FULL_ERROR=1

experiment="STL10_dist_variation_recpred"
notes="
**Goal**: Different distortions on STL10: iNCE,iVAE,VAE, predicted on reconstructions with Resnet18, ca. 100 runs for each config
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
architecture@predictor=resnet18
data@data_feat=stl10unlabeled
data@data_pred=stl10_aug
rate=H_hyper
trainer.max_epochs=100
+update_trainer_pred.max_epochs=100
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
"

# sweeping arguments
kwargs_hypopt_ivae="
distortion=vae,ivae
featurizer=neural_rec
architecture@encoder=resnet18
hydra.sweeper.n_trials=200
hydra.sweeper.n_jobs=200
monitor_direction=[minimize,minimize]
monitor_return=[test/pred/err,test/comm/rate]
distortion.factor_beta=1.0
data_feat.kwargs.batch_size=tag(log,int(interval(32,128)))
encoder.z_dim=tag(log,int(interval(32,512)))
featurizer.loss.beta_anneal=linear,constant
featurizer.loss.beta=tag(log,interval(1e-8,1e0))
distortion.factor_beta=tag(log,interval(1e-5,100))
rate.kwargs.warmup_k_epoch=int(interval(0,3))
rate.kwargs.invertible_processing=null,diag,psd
optimizer@optimizer_feat=Adam,AdamW
optimizer_feat.kwargs.weight_decay=tag(log,interval(1e-8,5e-4))
optimizer_feat.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer_feat.kwargs.is_lars=true,false
optimizer@optimizer_coder=SGD_likeadam,Adam
optimizer_coder.kwargs.weight_decay=tag(log,interval(1e-7,5e-4))
optimizer_coder.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_feat=cosine,expdecay100,expdecay1000,plateau_quick,plateau
scheduler@scheduler_coder=cosine_restart,expdecay100,plateau_quick,unifmultistep100
seed=0,1,2,3,4
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam,AdamW
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,expdecay1000,unifmultistep100
"

# sweeping arguments
kwargs_hypopt_jpeg="
featurizer=jpeg++,png,webp++
evaluation.featurizer.is_evaluate=False
featurizer.quality=1,3,5,10,20,30,40,70,95
hydra.sweeper.n_trials=300
hydra.sweeper.n_jobs=300
monitor_direction=[minimize]
monitor_return=[test/pred/err]
seed=0,1,2,3,4
data_pred.kwargs.batch_size=tag(log,int(interval(32,64)))
optimizer@optimizer_pred=SGD_likeadam,Adam
optimizer_pred.kwargs.weight_decay=tag(log,interval(1e-8,1e-4))
optimizer_pred.kwargs.lr=tag(log,interval(1e-4,1e-3))
scheduler@scheduler_pred=cosine,plateau_quick,cosine_restart,expdecay100,unifmultistep100
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_jpeg $kwargs_dep -m &

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_hypopt_ivae $kwargs_dep -m &

    sleep 7
  done
fi