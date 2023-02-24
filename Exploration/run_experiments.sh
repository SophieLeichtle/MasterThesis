#!/bin/bash

configs=(beechwood)
methods=('random' 'euclid' 'simple' 'visible' 'info' 'fusion')
iters=1

for config in "${configs[@]}"; do
    for method in "${methods[@]}"; do
        for ((i=0; i<iters; ++i)); do
            python explore_rtrrtstar.py -c "$config" -m "$method"
        done
    done
done