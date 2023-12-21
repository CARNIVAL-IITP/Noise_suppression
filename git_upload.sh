#!/bin/bash
PATH1='/home/user/share/national/Noise_suppression/2023'
for file in $PATH1/*
do
    filename=${file}
    echo "2023/${filename##/*/}"
    git add ${file}
    git commit -am "${filename##/*/}"
    git push origin main
    
done
