#!/usr/bin/env bash

experiment=$prfx"banana_ratedistortion"
notes="
**Goal**: Run rate distortion curve for banana distribution
**Hypothesis**: Should be close to the estimated optimal rate distortion curve
"

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`

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
loss.beta=0.0001,0.001,0.01,0.1,1,10,100
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
seed=1,2,3,4,5,6,7,8,9
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
# note: instead of using "ivae" and "vae" on the three augmented datasets (which gives 6 models), we run "ivae" on the 
#       three augmented + an unaugmented data (which is equivalent to running "vae" on the augmented one) => 4 models.
kwargs_multi="
data=bananaXtrnslt,bananaRot,bananaYtrnslt,banana
" 

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
  done
fi

#TODO plotting pipeline