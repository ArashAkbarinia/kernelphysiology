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

indir="/home/arash/Software/repositories/chesscnn/data/pgn/splittedgames/"
outdir="/home/arash/Software/repositories/chesscnn/data/fen/"

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

files=$indir"*.pgn"
for f in $files; 
do 
  echo "Processing $f ...";
  pgnname=$(basename $f)
  pgnname=${pgnname%".pgn"}
  fenname0="$outdir$pgnname"".fen"
  fenname1="$outdir$pgnname""p.fen"
  gnuchess -m <<EOD
pgnreplay $f
last
epdsave $fenname0
p
epdsave $fenname1
quit
EOD
done

