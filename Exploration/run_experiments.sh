#!/bin/bash

configs=('beechwood' 'wainscott')
methods=('fusion')
iters=10

for ((i=0; i<iters; ++i)); do
    for config in "${configs[@]}"; do
        for method in "${methods[@]}"; do
            python search_navgraph.py -c "$config" -m "$method"
        done
    done
done