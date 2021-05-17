#!/usr/bin/env bash

experiment="banana_viz_vae_square"
notes="
**Goal**: Run VAE and IVAE on banana distributions to get nice figures.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

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
featurizer.loss.is_square=true
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
featurizer.loss.beta=1e-2,7e-3
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "encoder.z_dim=1 distortion=ivae"  "encoder.z_dim=2 distortion=vae" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3

  done
fi

wait 

if [ "$is_plot_only" = true ] ; then

  #for featurizer
  col_val_subset=""
  python aggregate.py \
        experiment=$experiment  \
        $col_val_subset \
        agg_mode=[summarize_metrics]

  for kwargs_dep in  "encoder.z_dim=1 distortion=ivae" "encoder.z_dim=2 distortion=vae" 
    do

      col_val_subset=""
      python load_pretrained.py \
            load_pretrained.experiment=$experiment  \
            $col_val_subset \
            $kwargs  \
            +load_pretrained.collect_data.is_force_cpu=False \
            trainer.gpus=1 \
            $kwargs_multi \
            $kwargs_dep \
            load_pretrained.mode=[codebook_plot,maxinv_distribution_plot] \
            -m

      wait
    done
  fi

  done
fi
