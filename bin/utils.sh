#!/usr/bin/env bash

add_kwargs=""
prfx=""
time="1440" # 1 day
is_plot_only=false
server=""
mode=""
main="main.py"


# MODE ?
while getopts ':s:p:m:t:v:a:c:' flag; do
  case "${flag}" in
    s )
      server="${OPTARG}"
      add_kwargs="${add_kwargs} server=$server"
      echo "$server server ..."
      ;;
    c )
      id="${OPTARG}"
      mode="continue"
      add_kwargs="${add_kwargs} +mode=continue continue_job=${id}"
      echo "Continuing job $id (Only if slurm) ..."
      ;; 
    m ) 
      mode="${OPTARG}"
      add_kwargs="${add_kwargs} +mode=$mode"
      echo "$mode mode ..."

      if  [[ "$mode" != "cpu" ]]; then
        prfx="${mode}_"
        time="60"
      fi

      # overwrite max_epochs with the one from the mode
      if  [[ "$mode" == "dev" || "$mode" == "nano" ]]; then
        add_kwargs="${add_kwargs} trainer.max_epochs=2"
      fi
      
      ;;
    v ) 
      is_plot_only=true
      prfx=${OPTARG}
      echo "Visualization/plotting only ..."
      ;;
    p ) 
      main="parallel.py"
      add_kwargs="${add_kwargs} +parallel=${OPTARG}"
      echo "Parallel=${OPTARG} ..."
      ;;
    t ) 
      time=${OPTARG}
      echo "Time ${OPTARG} minutes"
      ;;
    a ) 
      add_kwargs="${add_kwargs} ${OPTARG}"
      echo "Adding ${OPTARG}"
      ;;
    \? ) 
      echo "Usage: "$name".sh [-stpmvac]" 
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done


if  [[ "$mode" == "dev" || "$mode" == "test" || "$mode" == "debug" || "$mode" == "nano" ]]; then
  case "$server" in
    learnfair) 
      add_kwargs="${add_kwargs} hydra.launcher.partition=dev"
      ;;
    vector) 
      add_kwargs="${add_kwargs} hydra.launcher.partition=interactive hydra.launcher.additional_parameters.qos=nopreemption"
      ;;
    qvector) 
      add_kwargs="${add_kwargs} hydra.launcher.partition=interactive hydra.launcher.additional_parameters.qos=nopreemption"
      ;;
  esac
fi

experiment="${prfx}""$experiment"

if [[ "$is_plot_only" = false && "$mode" != "continue" ]] ; then
  results="results/exp_$experiment"
  if [ -d "$results" ]; then

    echo -n "$results exist. Should I delete it (y/n) ? "
    read answer

    if [ "$answer" != "${answer#[Yy]}" ] ;then
        echo "Deleted $results"
        rm -rf $results
    fi
  fi  
fi