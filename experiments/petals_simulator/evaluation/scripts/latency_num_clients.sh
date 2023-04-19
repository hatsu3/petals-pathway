#!/bin/bash


conda activate petals


for dir in $(find $1 -type d -regex ".*$2.*" -print); do
    python3 evaluation/scripts/overall_average.py "$dir" >> evaluation/data/througput.csv
done

