#!/bin/bash
#Train an LM using Modified Kneser-Ney smoothing
# via the Google NGramLibrary

if [ $# -ne 2 ]
then
    echo "SYNTAX: `basename ${0}` <CORPUS> <ABS|KN|MKN>"
    exit
fi

bins=3
method="kneser_ney"
if [ "${2}" == "KN" ]
then
    bins=1
fi
if [ "${2}" == "ABS" ]
then
    bins=1
    method="absolute"
fi
if [ "${2}" == "ML" ]
then
    method="unsmoothed"
fi

ngramsymbols < ${1} > Gtrain.syms
farcompilestrings --symbols=Gtrain.syms --keep_symbols=1 ${1} > Gtrain.far
ngramcount --order=3 Gtrain.far > Gtrain.cnts
ngrammake --v=3 --method=${method} --bins=${bins} Gtrain.cnts > Gtrain.mod
ngramprint --ARPA Gtrain.mod Gtrain.arpa