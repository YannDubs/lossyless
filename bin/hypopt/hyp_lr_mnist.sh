#!/usr/bin/env bash

experiment="hyp_lr_mnist"
notes="
**Goal**: Hyperparameter tuning of the learning rate and learning rate of the encoder
**Plot**: RD curve for the three encoder
"

# what rate to use ? use the best encoder in terms of test performance
# what encoder to use ? use the best encoder in terms of test performance
# which distortion to use ? use the best distortion in terms of test performance

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment 
timeout=$time
rate=H_factorized
data=mnist
distortion=ivae
encoder=cnn
seed=1
optimizer_coder.scheduler.name='expdecay'
logger.wandb.project=hypopt
$add_kwargs
"

# use lr 0.001
# use epochs 200 in real and 100 when need quick
# use decay factor 100
# optimizer_coder.scheduler.decay_factor 100
# optimizer coder lr 0.003 

# every arguments that you are sweeping over
kwargs_multi="
optimizer.lr=0.0001,0.0003,0.001,0.003
optimizer.scheduler.decay_factor=10,100,1000
optimizer_coder.lr=0.0003,0.001,0.003
+optimizer_coder.scheduler.decay_factor=2,10,100
trainer.max_epochs=50,100,200
" 





if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
    sleep 3
    
  done
fi

#TODO plotting pipeline