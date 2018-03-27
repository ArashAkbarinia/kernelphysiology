#!/bin/bash

# convert a pgn file to fen for last position and one move before

echo "Generating fen files!"

indir=$1
outdir=$2

echo "Input directory $indir"
echo "Output directory $outdir"

files=$indir*.pgn
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

