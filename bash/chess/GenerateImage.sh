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

indir="/home/arash/Software/repositories/chesscnn/data/fen/"
outdir="/home/arash/Software/repositories/chesscnn/data/images/rotatedorg/"

if [ $1 = "white" ]; then
  outdir=$outdir"white/"
  indir=$indir"white/"
elif [ $1 = "black" ]; then
  outdir=$outdir"black/"
  indir=$indir"black/"
else
  outdir=$outdir"draw/"
  indir=$indir"draw/"
fi

echo "Input directory $indir"
echo "Output directory $outdir"

files=$indir"*.fen"
for f in $files; 
do 
  echo "Processing $f ...";
  fenname=$(basename $f)
  fenname=${fenname%".fen"}
  pngname="$outdir$fenname.png"
  /home/arash/Software/binaries/fen2ppm-0.1.0/fen2ppm $f | pnmtopng >$pngname
done

