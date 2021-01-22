#!/usr/bin/env bash

add_kwargs=""
prfx=""
time="2880" #2 days
is_plot_only=false
server=""
mode=""


# MODE ?
while getopts ':s:p:m:' flag; do
  case "${flag}" in
    s )
      server="${OPTARG}"
      add_kwargs="${add_kwargs} server=$server"
      echo "$server server ..."
      case "$server" in
        learnfair) 
          add_kwargs="${add_kwargs} hydra/launcher=submitit_slurm"
          ;;
        vector) 
          add_kwargs="${add_kwargs} hydra/launcher=submitit_slurm"
          ;;
        qvector) 
          add_kwargs="${add_kwargs} hydra/launcher=submitit_slurm"
          ;;
        local) 
          add_kwargs="${add_kwargs} hydra/launcher=submitit_local"
          ;;
        esac
      ;;
    m ) 
      mode="${OPTARG}"
      add_kwargs="${add_kwargs} +mode=$mode"
      prfx="$mode_"
      time="60"
      echo "$mode mode ..."
      ;;
    p ) 
      is_plot_only=true
      prfx=${OPTARG}
      echo "Plotting only ..."
      ;;
    \? ) 
      echo "Usage: "$name".sh [-spm]" 
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done


if  [[ "$mode" == "dev" || "$mode" == "test" || "$mode" == "debug" ]]; then
  case "$server" in
    learnfair) 
      add_kwargs="${add_kwargs} hydra.launcher.partition=dev"
      ;;
    vector) 
      add_kwargs="${add_kwargs} hydra.launcher.partition=interactive +hydra.launcher.additional_parameters.qos=nopreemption"
      ;;
    qvector) 
      add_kwargs="${add_kwargs} hydra.launcher.partition=interactive +hydra.launcher.additional_parameters.qos=nopreemption"
      ;;
  esac
fi

experiment="$prfx""$experiment"

results="results/$experiment"
if [ -d "$results" ]; then

  echo -n "$results exist. Should I delete it (y/n) ? "
  read answer

  if [ "$answer" != "${answer#[Yy]}" ] ;then
      echo "Deleted $results"
      rm -rf $results
  fi
fi  