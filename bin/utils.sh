#!/usr/bin/env bash

add_kwargs=""
prfx=""
time="2880" #2 days
is_plot_only=false


# MODE ?
while getopts ':s:p:m:' flag; do
  case "${flag}" in
    s )
      add_kwargs="${add_kwargs} server=${OPTARG}"
      echo "${OPTARG} server ..."
      case "${OPTARG}" in
        learnfair) 
          add_kwargs="${add_kwargs} hydra/launcher=submitit_slurm"
          ;;
        vector) 
          add_kwargs="${add_kwargs} hydra/launcher=submitit_slurm"
          ;;
        local) 
          add_kwargs="${add_kwargs} hydra/launcher=submitit_local"
          ;;
        esac
      ;;
    m ) 
      add_kwargs="${add_kwargs} +mode=${OPTARG}"
      prfx="${OPTARG}_"
      time="60"
      echo "${OPTARG} mode ..."
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