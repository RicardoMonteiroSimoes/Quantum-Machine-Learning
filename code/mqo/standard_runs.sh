#!/usr/bin/env bash

clear

if ! [ -d runs ]; then
    mkdir runs
fi

circuit=( hcs hcsx hcsh )
for i in "${circuit[@]}"
do
    echo doing circuit "$i"
	python static_qc_solver.py -s 5000 -sh 10000 -c "$i" -n runs/"$i"/ -pc
done