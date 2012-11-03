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


class ModKNSmoother( ):
    """
      Stand-alone python implementation of Fixed Modified Kneser-Ney discounting.

      Intended for educational purposes, this should produce results identical
       to Google NGramLibrary tools with ngrammake --bins=3.  See the included 
       run-NGramLibrary.sh script to train a model for comparison.
    
      WARNING: This may be slow for very large corpora.

    """

    def __init__( self, order=3, sb="<s>", se="</s>" ):
        self.sb        = sb
        self.se        = se
        self.order     = order
        self.ngrams    = NGramStack(order=order)
        self.denominators = [ defaultdict(float) for i in xrange(order-1) ]
        self.numerators   = [ defaultdict(float) for i in xrange(order-1) ]
        #Modified Kneser-Ney requires that we track the individual N_i
        # in contrast to Kneser-Ney, which just requires the sum-total.
        self.nonZeros     = [ defaultdict(lambda: defaultdict(float)) for i in xrange(order-1) ]
        self.CoC          = [ [ 0.0 for j in xrange(4) ] for i in xrange(order) ]
        self.discounts    = [ [ 0.0 for j in xrange(3) ] for i in xrange(order-1) ]
        self.UD = 0.
        self.UN = defaultdict(float)

    def _compute_counts_of_counts( self ):
        """
          Compute counts-of-counts (CoC) for each N-gram order.
          Only CoC<=4 are relevant to the computation of
          either ModKNFix or KNFix.
        """

        for k in self.UN:
            if self.UN[k] <= 4:
                self.CoC[0][int(self.UN[k]-1)] += 1.

        for i,dic in enumerate(self.numerators):
            for k in dic:
                if dic[k]<=4:
                    self.CoC[i+1][int(dic[k]-1)] += 1.
        return

    def _compute_discounts( self ):
        """
          Compute the discount parameters. Note that unigram counts
          are not discounted in either FixKN or FixModKN.

          ---------------------------------
          Fixed Modified Kneser-Ney: FixModKN
          ---------------------------------
          This is the solution proposed by Chen&Goodman '98

             Y    = N_1 / (N_1 + 2*N_2)
             D_1  = 1 - 2*Y * (N_2 / N_1)
             D_2  = 2 - 3*Y * (N_3 / N_2)
             D_3+ = 3 - 4*Y * (N_4 / N_3)

          where N_i again refers to the number of N-grams that appear
          exactly 'i' times in the training data.  The D_i refer to the
          counts-of-counts for the current N-gram.  That is, if the 
          current N-gram, 'a b c' was seen exactly two times in the 
          training corpus, then discount D_2 would be applied.

        """

        for o in xrange(self.order-1):
            Y = self.CoC[o+1][0] / (self.CoC[o+1][0]+2*self.CoC[o+1][1])
            #Compute all the D_i based on the formula
            for i in xrange(3):
                if self.CoC[o+1][i]>0:
                    self.discounts[o][i] = (i+1) - (i+2)*Y * (self.CoC[o+1][i+1]/self.CoC[o+1][i])
                else:
                    self.discounts[o][i] = (i+1)

        return 

    def _get_discount( self, order, ngram ):
        """
          Compute the discount mass for this N-gram, based on 
           the precomputed D_i and individual N_i.
        """

        c  = [0.0, 0.0, 0.0]
                    
        for key in self.nonZeros[order][ngram]:
            if int(self.nonZeros[order][ngram][key])==1:
                c[0] += 1.
            elif int(self.nonZeros[order][ngram][key])==2:
                c[1] += 1.
            else:
                c[2] += 1.

        #Compute the discount mass by summing over the D_i*N_i 
        d = sum([ self.discounts[order][i]*c[i] for i in xrange(len(c)) ])
        return d

    def kneser_ney_from_counts( self, arpa_file ):
        """
          Train the KN-discount language model from an ARPA format 
          file containing raw count data.  This can be generated with,
            $ ./SimpleCount.py --train train.corpus -r > counts.arpa

        """

        m_ord = c_ord = 0

        for line in open(arpa_file, "r"):
            ngram, count = line.strip().split("\t")
            count = float(count)
            ngram = ngram.split(" ")
            if len(ngram)==2:
                self.UD += 1.0

            if len(ngram)==2:
                self.UN[" ".join(ngram[1:])] += 1.0
                #Nonzeros based on suffixes
                if ngram[0]==self.sb:
                    self.nonZeros[len(ngram)-2][" ".join(ngram[:-1])][ngram[-1]] += count
                    self.numerators[len(ngram)-2][" ".join(ngram)] += count
                    self.denominators[len(ngram)-2][" ".join(ngram[:-1])] += count

            if len(ngram)>2 and len(ngram)<self.order:
                self.numerators[len(ngram)-3][" ".join(ngram[1:])] += 1.0
                self.denominators[len(ngram)-3][" ".join(ngram[1:-1])] += 1.0
                self.nonZeros[len(ngram)-3][" ".join(ngram[1:-1])][ngram[-1]] += 1.0
                if ngram[0]==self.sb:
                    self.numerators[len(ngram)-2][" ".join(ngram)] += count
                    self.denominators[len(ngram)-2][" ".join(ngram[:-1])] += count
                    self.nonZeros[len(ngram)-2][" ".join(ngram[:-1])][ngram[-1]] += count

            if len(ngram)==self.order:
                self.numerators[len(ngram)-3][" ".join(ngram[1:])] += 1.0
                self.numerators[len(ngram)-2][" ".join(ngram)] = count
                self.denominators[len(ngram)-3][" ".join(ngram[1:-1])] += 1.0
                self.denominators[len(ngram)-2][" ".join(ngram[:-1])] += count
                self.nonZeros[len(ngram)-3][" ".join(ngram[1:-1])][ngram[-1]] += 1.0
                self.nonZeros[len(ngram)-2][" ".join(ngram[:-1])][ngram[-1]] += count

        self._compute_counts_of_counts ( )
        self._compute_discounts( )

        #self._print_raw_counts( )
        return

    def kneser_ney_discounting( self, training_file ):
        """
          Iterate through the training data using a FIFO stack or 
           'window' of max-length equal to the specified N-gram order.

          Each time a new word is pushed onto the N-gram stack call
           the _kn_recurse() subroutine to increment the N-gram 
           contexts in the current window / on the stack. 

          If pushing a word onto the stack makes len(stack)>max-order, 
           then the word at the bottom (stack[0]) is popped off.
        """

        for line in open(training_file,"r"):
            #Split the current line into words.
            words = re.split(r"\s+",line.strip())

            #Push a sentence-begin token onto the stack
            self.ngrams.push(self.sb)

            for word in words:
                #Get the current 'window' of N-grams
                ngram = self.ngrams.push(word)

                #Now count all N-grams in the current window
                #These will be of span <= self.order
                self._kn_recurse( ngram, len(ngram)-2 )

            #Now push the sentence-end token onto the stack
            ngram = self.ngrams.push(self.se)
            self._kn_recurse( ngram, len(ngram)-2 )

            #Clear the stack for the next sentence
            self.ngrams.clear()
        self._compute_counts_of_counts ( )
        self._compute_discounts( )

        return


    def print_raw_counts( self ):
        """
          Convenience function for sanity checking the history counts.
        """
        #print "NUMERATORS:"
        #for key in sorted(self.UN.iterkeys()):
        #    print " ", key, self.UN[key]
        #for o in xrange(len(self.numerators)):
        #    print "ORD",o
        #    for key in sorted(self.numerators[o].iterkeys()):
        #        print " ", key, self.numerators[o][key]
        #print "DENOMINATORS:"
        #print self.UD
        #for o in xrange(len(self.denominators)):
        #    print "DORD", o
        #    for key in sorted(self.denominators[o].iterkeys()):
        #        print " ", key, self.denominators[o][key]
        print "NONZEROS:"
        for o in xrange(len(self.nonZeros)):
            print "ZORD", o
            for denom in sorted(self.nonZeros[o].iterkeys()):
                print " Den:", denom
                for key in sorted(self.nonZeros[o][denom].iterkeys()):
                    print "   ", key, self.nonZeros[o][denom][key]


    def _kn_recurse( self, ngram_stack, i ):
        """
         Kneser-Ney discount calculation recursion.
        """

        if i==-1 and ngram_stack[0]==self.sb:
            return

        o     = len(ngram_stack)
        numer = " ".join(ngram_stack[o-(i+2):])
        denom = " ".join(ngram_stack[o-(i+2):o-1])
        self.numerators[  i][numer] += 1.
        self.denominators[i][denom] += 1.
 
        #For Modified Kneser-Ney we need to track 
        # individual nonZeros based on their suffixes
        self.nonZeros[i][denom][ngram_stack[-1]] += 1.
        if self.numerators[i][numer]==1.:
            if i>0:
                self._kn_recurse( ngram_stack, i-1 )
            else:
                #The <s> (sentence-begin) token is
                # NOT counted as a unigram event
                if not ngram_stack[-1]==self.sb:
                    self.UN[ngram_stack[-1]] += 1.
                    self.UD += 1.
        return  
        
    def print_ARPA( self ):
        """
          Print the interpolated Kneser-Ney LM out in ARPA format,
           computing the interpolated probabilities and back-off
           weights for each N-gram on-demand.  The format:
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
        """

        #Handle the header info
        print "\\data\\"
        print "ngram 1=%d" % (len(self.UN)+1)
        for o in xrange(0,self.order-1):
            print "ngram %d=%d" % (o+2,len(self.numerators[o]) )

        #Handle the Unigrams
        print "\n\\1-grams:"
        d    = self._get_discount( 0, self.sb )
        #ModKN discount
        lmda = d / self.denominators[0][self.sb]
        print "-99.00000\t%s\t%0.7f"   % ( self.sb, log(lmda, 10.) )

        for key in sorted(self.UN.iterkeys()):
            if key==self.se:
                print "%0.7f\t%s\t-99"   % ( log(self.UN[key]/self.UD, 10.), key )
                continue

            d    = self._get_discount( 0, key )
            #ModKN discount
            lmda = d / self.denominators[0][key]
            print "%0.7f\t%s\t%0.7f" % ( log(self.UN[key]/self.UD, 10.), key, log(lmda, 10.) )

        #Handle the middle-order N-grams
        for o in xrange(0,self.order-2):
            print "\n\\%d-grams:" % (o+2)
            for key in sorted(self.numerators[o].iterkeys()):
                if key.endswith(self.se):
                    #No back-off prob for N-grams ending in </s>
                    prob = self._compute_interpolated_prob( key )
                    print "%0.7f\t%s" % ( log(prob, 10.), key )
                    continue
                d = self._get_discount( o+1, key )
                #Compute the back-off weight
                #ModKN discount
                lmda  = d / self.denominators[o+1][key]
                #Compute the interpolated N-gram probability
                prob = self._compute_interpolated_prob( key )
                print "%0.7f\t%s\t%0.7f" % ( log(prob, 10.), key, log(lmda, 10.))

        #Handle the N-order N-grams
        print "\n\\%d-grams:" % (self.order)
        for key in sorted(self.numerators[self.order-2].iterkeys()):
            #Compute the interpolated N-gram probability
            prob = self._compute_interpolated_prob( key )
            print "%0.7f\t%s" % ( log(prob, 10.), key )

        print "\n\\end\\"
        return


    def _compute_interpolated_prob( self, ngram ):
        """
          Compute the interpolated probability for the input ngram.
          Cribbing the notation from the SRILM webpages,

             a_z    = An N-gram where a is the first word, z is the 
                       last word, and "_" represents 0 or more words in between.
             p(a_z) = The estimated conditional probability of the 
                       nth word z given the first n-1 words (a_) of an N-gram.
             a_     = The n-1 word prefix of the N-gram a_z.
             _z     = The n-1 word suffix of the N-gram a_z.

          Then we have, 
             f(a_z) = g(a_z) + bow(a_) p(_z)
             p(a_z) = (c(a_z) > 0) ? f(a_z) : bow(a_) p(_z)

          The ARPA format is generated by writing, for each N-gram
          with 1 < order < max_order:
             p(a_z)    a_z   bow(a_z)

          and for the maximum order:
             p(a_z)    a_z

          special care must be taken for certain N-grams containing
           the <s> (sentence-begin) and </s> (sentence-end) tokens.  
          See the implementation for details on how to do this correctly.

          The formulation is based on the seminal Chen&Goodman '98 paper.

          SRILM notation-cribbing from:
             http://www.speech.sri.com/projects/srilm/manpages/ngram-discount.7.html
        """
        probability = 0.0
        ngram_stack = ngram.split(" ")
        probs       = [ 1e-99 for i in xrange(len(ngram_stack)) ]
        o           = len(ngram_stack)

        if not ngram_stack[-1]==self.sb:
            probs[0] = self.UN[ngram_stack[-1]] / self.UD

        for i in xrange(o-1):
            dID = " ".join(ngram_stack[o-(i+2):o-1])
            nID = " ".join(ngram_stack[o-(i+2):])
            if dID in self.denominators[i]:
                count = int(self.numerators[i][nID])
                d_i   = min(count, 3)
                probs[i+1] = (count-self.discounts[i][d_i-1])/self.denominators[i][dID]

                #This break-down takes the following form:
                #  probs[i+1]:  The interpolated N-gram probability, p(a_z)
                #  d:           The discount mass for a_: \Sum_i D_i*N_i
                #  lmda:        The un-normalized 'back-off' weight, bow(a_)
                #  probs[i]:    The next lower-order, interpolated N-gram 
                #               probability corresponding to p(_z)
                #ModKN discount
                d = self._get_discount( i, dID )
                lmda        = d / self.denominators[i][dID]
                probs[i+1]  = probs[i+1] + lmda * probs[i]
                probability = probs[i+1]

        if probability == 0.0:
            #If we still have nothing, return the unigram probability
            probability = probs[0]

        return probability

if __name__=="__main__":
    import sys, argparse

    example = """%s --train train.corpus""" % sys.argv[0]
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument('--train',    "-t", help="The text corpus to be used to train the LM.", required=True )
    parser.add_argument('--order',    "-o", help="The maximum N-gram order (3).", required=False, default=3, type=int )
    parser.add_argument('--sb',       "-b", help="The sentence-begin token (<s>).", required=False, default="<s>" )
    parser.add_argument('--se',       "-e", help="The sentence-end token (</s>).", required=False, default="</s>" )
    parser.add_argument('--counts',   "-c", help="The input file contains raw counts.", action="store_true", default=False )
    parser.add_argument('--verbose',  "-v", help="Verbose mode.", action="store_true", default=False )
    args = parser.parse_args()

    if args.verbose:
        for attr, value in args.__dict__.iteritems():
            print attr, "=", value
    lms = ModKNSmoother( order=args.order, sb=args.sb, se=args.se )
    if args.counts:
        lms.kneser_ney_from_counts( args.train )
    else:
        lms.kneser_ney_discounting( args.train )
    lms.print_ARPA( )
    #lms.print_raw_counts( )
