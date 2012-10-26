#!/bin/bash
#Train an LM using Modified Kneser-Ney smoothing
# via the Google NGramLibrary

if [ $# -ne 2 ]
then
    echo "SYNTAX: `basename ${0}` <CORPUS> <KN|MKN>"
    exit
fi

bins=3
if [ "${2}" == "KN" ]
then
    bins=1
fi

ngramsymbols < ${1} > Gtrain.syms
farcompilestrings --symbols=Gtrain.syms --keep_symbols=1 ${1} > Gtrain.far
ngramcount --order=3 Gtrain.far > Gtrain.cnts
ngrammake --v=3 --method=kneser_ney --bins=${bins} Gtrain.cnts > Gtrain.mod
ngramprint --ARPA Gtrain.mod Gtrain.arpa