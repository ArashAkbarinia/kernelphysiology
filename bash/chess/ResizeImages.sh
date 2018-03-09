#!/bin/bash

# example of using arguments to a script
#echo "My first name is $1"
#echo "My surname is $2"
#echo "My surname is $3"
#echo "Total number of arguments is $#" 

#-Tr1-0
#-Tr0-1
#-Tr1/2

echo "Generating fen files!"

indir="/home/arash/Software/repositories/chesscnn/data/images/org/"
outdir128="/home/arash/Software/repositories/chesscnn/data/images/s128/"
outdir64="/home/arash/Software/repositories/chesscnn/data/images/s64/"

mkdir $outdir128
mkdir $outdir64

if [ $1 = "white" ]; then
  outdir128=$outdir128"white/"
  outdir64=$outdir64"white/"
  indir=$indir"white/"
elif [ $1 = "black" ]; then
  outdir128=$outdir128"black/"
  outdir64=$outdir64"black/"
  indir=$indir"black/"
else
  outdir128=$outdir128"draw/"
  outdir64=$outdir64"draw/"
  indir=$indir"draw/"
fi

mkdir $outdir128
mkdir $outdir64

echo "Input directory $indir"
echo "Output directory $outdir"

files=$indir"*.png"
for f in $files; 
do 
  echo "Processing $f ...";
  inname=$(basename $f)
  inname=${inname%".png"}

  outname64="$outdir64$inname.jpg"
  convert -resize 64x64 -quality 100 $f $outname64

  outname128="$outdir128$inname.jpg"
  convert -resize 128x128 -quality 100 $f $outname128
done

