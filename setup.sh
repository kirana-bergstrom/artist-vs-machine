#!/bin/bash

# pulls categories from categories.txt file
categories_file='categories.txt'

# creates needed data directories
mkdir -p data/raw
mkdir -p data/simplified
mkdir -p data/student_svgs
mkdir -p models

# downloads the relevant categorical data from google
while read line; do  
    URL="gs://quickdraw_dataset/full/raw/$line.ndjson"
    gsutil -m cp $URL data/raw
    i=$((i+1))  
done < $categories_file
