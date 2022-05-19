#!/usr/bin/env bash

clear

if ! [ -d runs/functionality_analysis ]; then
    mkdir runs/functionality_analysis/
fi

python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_1_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_2_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_3_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_4_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_5_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_6_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_7_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_8_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_9_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_10_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_11_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_12_"
python static_qc_solver.py -s 1024 -sh 10000 -c hcsx -n "runs/functionality_analysis/iteration_13_"
