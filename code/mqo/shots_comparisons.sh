#!/usr/bin/env bash

clear

if ! [ -d runs/shots_comparison ]; then
    mkdir runs/shots_comparison
fi
shots=( 1 10 25 50 75 100 150 200 350 500 1000 1500 2000 10000)
for i in "${shots[@]}"
do
    if ! [ -d runs/shots_comparison/shots"$i" ]; then
        mkdir runs/shots_comparison/shots"$i"
    fi
    echo doing circuit with "$i" shots
	python static_qc_solver.py -s 5000 -sh "$i" -c hcsx -n runs/shots_comparison/shots"$i"/ -pc
done