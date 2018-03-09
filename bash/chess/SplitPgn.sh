#!/bin/bash

# example of using arguments to a script
#echo "My first name is $1"
#echo "My surname is $2"
#echo "My surname is $3"
#echo "Total number of arguments is $#" 

#-Tr1-0
#-Tr0-1
#-Tr1/2

echo "Splitting the pgn files!"

outdir="/home/arash/Software/repositories/chesscnn/data/pgn/splittedgames/"

if [ $1 = "white" ]; then
  tr="-Tr1-0"
  outdir=$outdir"white/"
elif [ $1 = "black" ]; then
  tr="-Tr0-1"
  outdir=$outdir"black/"
else
  tr="-Tr1/2"
  outdir=$outdir"draw/"
fi

cd $outdir
echo "Output directory $outdir"

files="/home/arash/Software/repositories/chesscnn/data/pgn/orgdatasets/*.pgn"
for f in $files; 
do 
  echo "Processing $f ...";
  n=$(ls -1q $outdir"*.pgn" | wc -l)
  n=$((n + 1))
  echo "Number of files $n"
  /home/arash/Software/binaries/pgn-extract/pgn-extract $f -#1,$n $tr -bl20 --quiet
done

