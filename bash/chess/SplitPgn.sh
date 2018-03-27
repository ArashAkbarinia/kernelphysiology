#!/bin/bash

# extract single games from a database of chess games in pgn format

echo "Splitting the pgn files!"

outdir=$2

if [ $3 = "white" ]; then
  tr="-Tr1-0"
  outdir=$outdir"white/"
elif [ $3 = "black" ]; then
  tr="-Tr0-1"
  outdir=$outdir"black/"
else
  tr="-Tr1/2"
  outdir=$outdir"draw/"
fi

echo "Output directory $outdir"
cd $outdir

files=$1*.pgn
for f in $files; 
do 
  echo "Processing $f ...";
  n=$(ls -1q $outdir*.pgn | wc -l)
  n=$((n + 1))
  echo "Number of files $n"
  /home/arash/Software/binaries/pgn-extract/pgn-extract $f -#1,$n $tr "${@:4}" --quiet
done

