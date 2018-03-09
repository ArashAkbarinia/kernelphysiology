#!/bin/bash

echo "Generating a list of training and teting images!"

indir=$1
trainfile=$indir"train.txt"
testfile=$indir"test.txt"

rm $trainfile $testfile

declare -a arr=("white" "black" "draw")

for j in "${arr[@]}"
do
  if [ $j = "white" ]; then
    label=1
  elif [ $j = "black" ]; then
    label=2
  else
    label=3
  fi
  echo "$j"
  files=$indir$j
  for i in {1..30000}
  do
    f0=$files"/"$i".jpg" 
    f1=$files"/"$i"p.jpg"
    echo "$f0 $label" >> $trainfile
    echo "$f1 $label" >> $trainfile
  done
  for i in {30001..35000}
  do
    f0=$files"/"$i".jpg" 
    f1=$files"/"$i"p.jpg"
    echo "$f0 $label" >> $testfile
    echo "$f1 $label" >> $testfile
  done
done

