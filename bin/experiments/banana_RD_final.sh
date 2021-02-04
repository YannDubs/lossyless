#!/usr/bin/env bash

experiment=$prfx"banana_RD_final"
notes="
**Goal**: Run rate distortion curve for banana distribution when predicting representative
**Hypothesis**: Should be close to the estimated optimal rate distortion curve, and now can approximate it using differential entropies.
"

# e.g. command: bin/experiments/banana_RD_final.sh -s vector 

# parses special mode for running the script
source `dirname $0`/../utils.sh


# most of these arguments are chose so as to replicate Fig.1.b. from "Non linear Transform coding" paper. 
# See their code here: https://github.com/tensorflow/compression/blob/master/models/toy_sources/toy_sources.ipynb
# only differences (as far as I can tell):
# - use batch norm
# - hidden dim for MLPs is 1024 instead of 100
# - beta = 0.15 (for no invaraince) instead of 1
# - train batch size is 8192 instead of 1024
# - not using soft rounding

# Encoder
encoder_kwargs="
encoder=mlp
encoder.z_dim=2
encoder.arch_kwargs.complexity=null
+encoder.arch_kwargs.activation=Softplus
+encoder.arch_kwargs.hid_dim=1024
+encoder.arch_kwargs.n_hid_layers=2
+encoder.arch_kwargs.norm_layer=batchnorm
"

# Decoder
decoder_kwargs="
distortion.factor_beta=1
distortion.kwargs.arch=mlp
distortion.kwargs.arch_kwargs.complexity=null
+distortion.kwargs.arch_kwargs.activation=Softplus
+distortion.kwargs.arch_kwargs.hid_dim=1024
+distortion.kwargs.arch_kwargs.n_hid_layers=2
+distortion.kwargs.arch_kwargs.norm_layer=batchnorm
"

# Loss
rate_kwargs="
rate=H_factorized
rate.factor_beta=1
"

# Training / General
general_kwargs="
optimizer.lr=1e-3
optimizer.scheduler.name=MultiStepLR
+optimizer.scheduler.milestones=[50,75,87]
optimizer_coder.lr=1e-3
optimizer_coder.name=null
data=bananaRot
data.kwargs.batch_size=8192
data.kwargs.dataset_kwargs.length=1024000
data.kwargs.val_size=100000
data.kwargs.val_batch_size=16384
+data.kwargs.dataset_kwargs.decimals=null
trainer.max_epochs=100
trainer.precision=32
trainer.reload_dataloaders_every_epoch=True
evaluation.is_est_entropies=True
"

kwargs="
experiment=$experiment 
$general_kwargs
$encoder_kwargs
$decoder_kwargs
$rate_kwargs
timeout=${time}
$add_kwargs
"

kwargs_multi="
distortion=ivae,vae
loss.beta=0.000001,0.00001,0.0001,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,100,1000,10000
seed=1,2,3,4,5
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" 
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3
    
  done
fi

#TODO plotting pipeline