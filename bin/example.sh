#!/usr/bin/env bash

name=$prfx"debug"
notes="
**Goal**: Debugging all models at once (in parallel)
**Hypothesis**: No errors
"

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
name=$name 
+mode=debug
predictor.is_online_eval=False
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
encoder=cnn,mlp,resnet
distortion=ivib,ivae,ince,taskvib,vae,nce
rate=H_factorized,H_hyper,MI_unitgaussian,MI_vamp
data=mnist,bananaRot
" 

if [ "$is_plot_only" = false ] ; then
    # add kwargs if parameters have dependecies, so you cannot use Hydra's multirun (they will run in parallel in the background). 
    # e.g. `for kwargs1 in "a=1,2 b=3,4" "a=11,12 b=13,14" `
  for kwargs_dep in  ""
  do

    python main.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &
    
  done
fi

#TODO plotting pipeline