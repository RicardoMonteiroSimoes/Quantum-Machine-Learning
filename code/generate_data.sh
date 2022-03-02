#!/usr/bin/env bash
echo "Generating datasets"
if ! [ -d "datasets" ]; then
    mkdir datasets
fi
python generate_datasets.py -n datasets/two_features_two_classes_blobs -m blobs -ns 0.2 -f 2 -c 2 -s 300 -ps -r 696969
python generate_datasets.py -n datasets/two_features_two_classes_gaussian -m gaussian -ns 0.2 -f 2 -c 2 -s 300 -ps -r 696969
python generate_datasets.py -n datasets/two_features_two_classes_moons -m moons -ns 0.15 -f 2 -c 2 -s 300 -ps -r 696969
python generate_datasets.py -n datasets/two_features_two_classes_circles -m circles -ns 0.05 -f 2 -c 2 -s 300 -ps -r 696969
python generate_datasets.py -n datasets/two_features_two_classes_classification -m classification -ncpc 1 -ninf 2 -nred 0 -nrep 0 -ns 0.2 -f 2 -c 2 -s 300 -ps -r 696969
python generate_datasets.py -n datasets/three_features_two_classes_classification -m classification -3d -ncpc 1 -ninf 3 -nred 0 -nrep 0 -ns 0.2 -f 3 -c 2 -s 300 -r 696969