#!/usr/bin/env bash

add_kwargs=""
prfx=""
run="0,1,2,3,4"
time="2880" #2 days
is_plot_only=false


# DEV MODE ?
while getopts ':dvtnsi:p:l' flag; do
  case "${flag}" in
    d ) 
      add_kwargs='+mode=debug' 
      time="10"
      prfx="dev_"
      run="0"
      echo "Debug mode ..."
      ;;
    v ) 
      add_kwargs='+mode=dev' 
      time="10"
      prfx="dev_"
      run="0"
      echo "Dev mode ..."
      ;;
    t ) 
      add_kwargs='+mode=test' 
      time="60"
      prfx="test_"
      run="0"
      echo "Test mode ..."
      ;;
    s ) 
      add_kwargs='datasize.max_epochs=100 +logger.wandb.tags="small"' 
      time="800"
      prfx="small_"
      run="0"
      echo "Small mode ..."
      ;;
    l ) 
      add_kwargs='datasize.max_epochs=300 +logger.wandb.tags="large"' 
      prfx="large_"
      run="0,1,2,3,4,5,6,7,8,9"
      echo "Large mode ..."
      ;;
    p ) 
      is_plot_only=true
      prfx=${OPTARG}
      echo "Plotting only ..."
      ;;
    i ) 
      arguments="hydra.launcher.partition=priority hydra.launcher.params.queue_parameters.slurm.comment=${OPTARG}"
      echo "Priority mode : ${OPTARG}..."
      ;;
    \? ) 
      echo "Usage: "$name".sh [-dvtnsipl]" 
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done

name="$prfx""$name"

results="results/$name"
if [ -d "$results" ]; then

  echo -n "$results exist. Should I delete it (y/n) ? "
  read answer

  if [ "$answer" != "${answer#[Yy]}" ] ;then
      echo "Deleted $results"
      rm -rf $results
  fi
fi  