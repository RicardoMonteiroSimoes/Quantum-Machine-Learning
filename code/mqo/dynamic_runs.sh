#!/usr/bin/env bash

clear

if ! [ -d runs ]; then
    mkdir runs
fi
if ! [ -d runs/dynamic ]; then
    mkdir runs/dynamic
fi


#Size must be one smaller than the total amount of problems
queries=( 2 2 2 2 2 2 3 3 3 3 )
queryplans=( "2 3" "3 2" "3 3" "4 2" "4 3" "4 4" "2 2 2" "2 3 2" "3 2 3" "3 3 3") 
for i in {0..11}
do
    echo doing problem "$i"
    if ! [ -d runs/dynamic/$i ]; then
        mkdir runs/dynamic/$i
    fi
	python dynamic_qc_solver.py -s 5000 -sh 10000 -n runs/dynamic/$i/ -pc -q ${queries[$i]} -qp ${queryplans[$i]}
done