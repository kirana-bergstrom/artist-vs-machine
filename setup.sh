#!/bin/bash

# pulls categories from categories.txt file
categories_file='categories.txt'

# creates needed data directories
mkdir -p data/raw

# downloads the relevant categorical data from google
while read line; do  
    URL="gs://quickdraw_dataset/full/raw/$line.ndjson"
    gsutil -m cp $URL data/raw
    i=$((i+1))  
done < $categories_file

# reduces and simplifies the data
NUM_SIMPLE=10000
python simple_data.py $NUM_SIMPLE
