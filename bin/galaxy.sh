#!/usr/bin/env bash

name=$prfx"galaxy"
notes="
Testing the API for galaxy dataset(s).
"

# parses special mode for running the script
source `dirname $0`/utils.sh $1
echo "source utils.sh complete"

# Encoder
encoder_kwargs="
encoder=mlp
encoder.z_dim=40
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
distortion=vae
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
logger.name=tensorboard
data=galaxy64
$general_kwargs
$encoder_kwargs
$decoder_kwargs
$rate_kwargs
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  " "
  do
    python main.py $kwargs $kwargs_multi $kwargs_dep -m &
  done
fi