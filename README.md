SimpleLM
========

Simple, standalone python classes for training statistical language models using several popular smoothing methods.

Also includes a script for evaluating LM sentence probabilities and retrieving individual NGram probabilities.

Example
=======

  ```$ ./SimpleModKN.py -t train.corpus -o 3```

  Sentence:
  ```$ ./evaluate-lm.py -m test.arpa -s "a b c d"```

  NGram:
  ```$ ./evaluate-lm.py -m test.arpa -s "a b c" -g```