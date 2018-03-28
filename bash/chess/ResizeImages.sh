#!/bin/bash

# resizing images of a folder

echo "Resizing images!"

indir=$1$3"/"
outdir=$2$3"/"

echo "Input directory $indir"
echo "Output directory $outdir"

mkdir $outdir

files=$indir"*.png"
for f in $files; 
do 
  echo "Processing $f ...";
  inname=$(basename $f)
  inname=${inname%".png"}

  outname64="$outdir$inname.jpg"
  convert -resize $4x$4 -quality 100 $f $outname64
done

