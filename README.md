SimpleLM
========

Simple, standalone python classes for training statistical language models using several popular smoothing methods.

Motivations
===========

The main motivation for these tools is to provide simple, standalone, python-based implementations of popular smoothing algorithms used in statistical language modeling.

Although there are many existing, open-source language modeling toolkits, most of these existing toolkits are designed for heavy use, and thus tend to sacrifice readability.  The connection between the theory and the implementation is often non-obvious.

Implementations
===============

The SimpleLM classes can be used to train arbitrary-order models, and correctly handle sentence-begin and sentence-end tokens, as well as several other non-obvious details.  The implementations are based on the seminal Chen & Goodman paper, 'An Empirical Study of Smoothing Techniques for Language Modeling', and are hopefully straightforward.  Although the classes are implemented in python, and not particularly optimized, they are surprisingly fast, especially for medium sized data sets.

Current Offerings
=================

Currently there are just two classes, 

  *  SimpleKN.py :   Implementation of Kneser-Ney smoothing.

  *  SimpleModKN.py : Implemenation of Modified Kneser-Ney smoothing

There is also a script, which relies on the Google NGramLibrary to train an LM for comparison,

  * run-NGramLibrary.sh : Train a KN or MKN model to compare the output with SimpleLM

Planned Additions
=================

  * Absolute discounting
 
  * Witten-Bell discounting


Example
=======

  ```$ ./SimpleModKN.py -t train.corpus -o 3```