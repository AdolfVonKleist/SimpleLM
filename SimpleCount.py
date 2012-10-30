#!/usr/bin/python
# Copyright (c) [2012-], Josef Robert Novak
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
#  modification, are permitted #provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright 
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above 
#    copyright notice, this list of #conditions and the following 
#    disclaimer in the documentation and/or other materials provided 
#    with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
# OF THE POSSIBILITY OF SUCH DAMAGE.
from collections import defaultdict
from NGramStack import NGramStack
from math import log
import re



class MLCounter( ):
    """
      Stand-alone python implementation of an simple Maximum Likelihood LM.

      This class simply counts NGrams in a training corpus and either,
        * Dumps the raw, log_10 counts into ARPA format
        * Computes an unsmoothed Maximum Likelihood LM 
    """

    def __init__( self, order=3, sb="<s>", se="</s>", raw=False, ml=False ):
        self.sb        = sb
        self.se        = se
        self.raw       = raw
        self.ml        = ml
        self.order     = order
        self.ngrams    = NGramStack(order=order)
        self.counts    = [ defaultdict(float) for i in xrange(order) ]

    def maximum_likelihood( self, training_file ):
        """
          Iterate through the training data using a FIFO stack or 
           'window' of max-length equal to the specified N-gram order.

          Each time a new word is pushed onto the N-gram stack call
           the _ml_count() subroutine to increment the N-gram counts.

          If pushing a word onto the stack makes len(stack)>max-order, 
           then the word at the bottom (stack[0]) is popped off.
        """

        for line in open(training_file,"r"):
            #Split the current line into words.
            words = re.split(r"\s+",line.strip())

            #Push a sentence-begin token onto the stack
            ngram = self.ngrams.push(self.sb)
            self._ml_count( ngram )

            for word in words:
                #Get the current 'window' of N-grams
                ngram = self.ngrams.push(word)

                #Now count all N-grams in the current window
                #These will be of span <= self.order
                self._ml_count( ngram )

            #Now push the sentence-end token onto the stack
            ngram = self.ngrams.push(self.se)
            self._ml_count( ngram )

            #Clear the stack for the next sentence
            self.ngrams.clear()
        
        return

    def _ml_count( self, ngram_stack ):
        """
          Just count NGrams.  The only slightly confusing thing here
          is the sentence-begin (<s>).  It does NOT count as a
          unigram event and thus does not contribute to the unigram tally.
          It IS however used as a history denominator.
        """

        #Iterate backwards through the stack
        for o in xrange(len(ngram_stack),0,-1):
            start = len(ngram_stack)-o
            self.counts[o-1][" ".join(ngram_stack[start:])] += 1.0

        return  
        
    def print_ARPA( self ):
        """
          Print the raw counts or ML LM out in ARPA format,
           ARPA format:
           ----------------------------
             \data\
             ngram 1=NUM_1GRAMS
             ngram 2=NUM_2GRAMS
             ...
             ngram N=NUM_NGRAMS (max order)
            
             \1-grams:
             p(a_z)  a_z  bow(a_z)
             ...
            
             \2-grams:
             p(a_z)  a_z  bow(a_z)
             ...
            
             \N-grams:
             p(a_z)  a_z
             ...

             \end\
           ----------------------------
          NOTE: Neither the ML model nor the raw counts
                will ever have a 'backoff weight'.
        """

        #Handle the header info
        print "\\data\\"
        for o in xrange(0,self.order):
            print "ngram %d=%d" % (o+1,len(self.counts[o]) )

        #Handle the Unigrams
        print "\n\\1-grams:"
        for key in sorted(self.counts[0].iterkeys()):
            if key==self.sb:
                if self.raw:
                    print "0.00000\t%s"   % ( self.sb )
                else:
                    print "-99.00000\t%s" % ( self.sb )
            else:
                if self.ml:
                    ml_prob = self.counts[0][key] / ( sum( [ self.counts[0][c] for c in self.counts[0].keys() ] ) - self.counts[0][self.sb] )
                    if self.raw:
                        print "%0.6f\t%s" % ( ml_prob, key )
                    else:
                        print "%0.6f\t%s" % ( log(ml_prob,10.), key )
                elif self.raw:
                    print "%0.6f\t%s" % (self.counts[0][key], key)
                else:
                    print "%0.6f\t%s" % (log(self.counts[0][key],10.), key)

        #Handle the middle-order N-grams
        for o in xrange(1,self.order):
            print "\n\\%d-grams:" % (o+1)
            for key in sorted(self.counts[o].iterkeys()):
                if self.ml:
                    hist = key[:key.rfind(" ")]
                    ml_prob = self.counts[o][key] / self.counts[o-1][hist]
                    if self.raw:
                        print "%0.6f\t%s" % ( ml_prob, key )
                    else:
                        print "%0.6f\t%s" % ( log(ml_prob,10.), key )
                elif self.raw:
                    print "%0.6f\t%s" % (self.counts[o][key], key)
                else:
                    print "%0.6f\t%s" % (log(self.counts[o][key],10.), key)

        print "\n\\end\\"
        return


if __name__=="__main__":
    import sys, argparse

    example = """%s --train train.corpus""" % sys.argv[0]
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument('--train',     "-t", help="The text corpus to be used to train the LM.", required=True )
    parser.add_argument('--order',     "-o", help="The maximum N-gram order (3).",   required=False, default=3, type=int )
    parser.add_argument('--sb',        "-b", help="The sentence-begin token (<s>).", required=False, default="<s>" )
    parser.add_argument('--se',        "-e", help="The sentence-end token (</s>).",  required=False, default="</s>" )
    parser.add_argument('--ml',        "-m", help="Compute the ML model.", action="store_true", default=False )
    parser.add_argument('--raw',       "-r", help="Output the raw counts, not log_10.", action="store_true", default=False )
    parser.add_argument('--verbose',   "-v", help="Verbose mode.", action="store_true", default=False )
    args = parser.parse_args()

    if args.verbose:
        for attr, value in args.__dict__.iteritems():
            print attr, "=", value
    lms = MLCounter( order=args.order, sb=args.sb, se=args.se, ml=args.ml, raw=args.raw )
    lms.maximum_likelihood( args.train )
    lms.print_ARPA( )
