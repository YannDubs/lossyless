#!/usr/bin/env bash

name=$prfx"banana"
notes="
**Goal**: Run banana models for plotting.
**Hypothesis**: When using some invariance the rate should go down for a similar distortion. Furthermore, the codebook should be reminiscent of the maximal invariant.
"

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
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
distortion=ivae
"


# Training / General
general_kwargs="
optimizer.lr=1e-3
optimizer.scheduler.name=MultiStepLR
+optimizer.scheduler.milestones=[50,75,87]
data.kwargs.batch_size=8192
data.kwargs.dataset_kwargs.length=1024000
data.kwargs.val_size=100000
data.kwargs.val_batch_size=16384
trainer.max_epochs=100
trainer.precision=32
trainer.reload_dataloaders_every_epoch=True
"

kwargs="
name=$name 
$general_kwargs
$encoder_kwargs
$decoder_kwargs
$rate_kwargs
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
" 


if [ "$is_plot_only" = false ] ; then
  # note: instead of using "ivae" and "vae" on the three augmented datasets (which gives 6 models), we run "ivae" on the 
  #       three augmented + an unaugmented data (which is equivalent to running "vae" on the augmented one) => 4 models.
  # using smaller beta for no invaraiance to be comparable
  for kwargs_dep in  "data=bananaRot,bananaXtrnslt,bananaYtrnslt loss.beta=0.3" "data=banana loss.beta=0.15"
  do

    python main.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
  done
fi

#TODO plotting pipeline