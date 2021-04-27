#!/usr/bin/env bash

experiment="banana_viz_vae"
notes="
**Goal**: Run VAE and IVAE on banana distributions to get nice figures.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh


# most of these arguments are chose so as to replicate Fig.1.b. from "Non linear Transform coding" paper. 
# See their code here: https://github.com/tensorflow/compression/blob/master/models/toy_sources/toy_sources.ipynb
# there are known diferences like:
# - use batch norm
# - hidden dim for MLPs is 1024 instead of 100
# - beta = 0.1 (for no invaraince) instead of 1
# - not using soft rounding
# - 200 epochs and scheduler
# - annealing beta

# Encoder
encoder_kwargs="
architecture@encoder=fancymlp
encoder.z_dim=2
"

# Distortion
distortion_kwargs="
distortion.factor_beta=1
architecture@distortion.kwargs=fancymlp
"
# like in their paper we are using softplus activation which gives slightly more smooth decision boundaries 

# Rate
rate_kwargs="
rate=H_factorized
rate.factor_beta=1
"

# Data
data_kwargs="
data@data_feat=banana_rot
trainer.reload_dataloaders_every_epoch=True
"

# Featurizer
general_kwargs="
is_only_feat=False
featurizer=neural_feat
optimizer@optimizer_feat=Adam
optimizer_feat.kwargs.lr=1e-3
scheduler@scheduler_feat=expdecay1000
optimizer@optimizer_coder=Adam
scheduler@scheduler_coder=expdecay100
optimizer_coder.kwargs.lr=3e-4
trainer.max_epochs=100
trainer.precision=32
architecture@predictor=mlp_probe
optimizer@optimizer_pred=Adam
scheduler@scheduler_pred=unifmultistep100
optimizer_pred.kwargs.lr=1e-3
"

kwargs="
logger.kwargs.project=banana
experiment=$experiment 
$encoder_kwargs
$distortion_kwargs
$rate_kwargs
$data_kwargs
$general_kwargs
timeout=${time}
$add_kwargs
"

kwargs_multi="
data@data_feat=banana_rot,banana_Xtrnslt,banana_Ytrnslt
distortion=ivae,vae
featurizer.loss.beta=0.07
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3

  done
fi

wait 

# for featurizer
# col_val_subset=""
# python aggregate.py \
#        experiment=$experiment  \
#        $col_val_subset \
#        agg_mode=[summarize_metrics]


kwargs_multi="
data@data_feat=banana_rot
distortion=ivae
featurizer.loss.beta=0.07
featurizer=neural_rec
"

col_val_subset=""
python load_pretrained.py \
       load_pretrained.experiment=$experiment  \
       $col_val_subset \
       $kwargs  \
       server=local \
       trainer.gpus=0 \
       $kwargs_multi \
       load_pretrained.mode=[codebook_plot,maxinv_distribution_plot] 