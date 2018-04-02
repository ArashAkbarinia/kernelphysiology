#!/bin/bash

# generates png image from the fen files

echo "Generating png files!"

indir=$1
outdir=$2
whowon=$3

echo "Input directory $indir"
echo "Output directory $outdir"

files=$indir"*.fen"
for f in $files; 
do 
  echo "Processing $f ...";
  fenname=$(basename $f)
  fenname=${fenname%".fen"}

  line=$(head -n 1 $f)

  stringarray=($line)

  islast="0"
  if [ ${fenname: -1} = 'p' ]; then
    islast="1"
  fi

  fenname=${fenname%"p"}
  txtname="$outdir${stringarray[1]}$whowon$islast/$fenname.txt"
  /home/arash/Software/repositories/kernelphysiology/cpp/src/fen2ppm/fen2txt $f >$txtname
done
