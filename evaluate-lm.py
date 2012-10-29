#!/usr/bin/python
from collections import defaultdict
import re

class LMTree( ):
    """
      A simple trie structure suitable for representing 
      a standard statistical language model.

      Each node stores a variety of useful information:

        * ngram:     The word ID for this node
        * prob:      The (log_10) probability for this node
        * bow:       The backoff weight for this node - if any
        * parent:    A pointer to the parent node - if any
        * children:  A hash of child-nodes - if any
                      - This encodes higher-order Ngrams

      Two public methods are also implemented:

        * add_child( ngram, prob=0.0, bow=0.0 )
            Create a new child node, an instance of LMTree, and
             add it to the hash of children for this node.

        * get_ngram_p( ngram, i=0 )
            Recursive method for retrieving the probability
             or backoff weight for an input ngram. 
    """
    def __init__( self, ngram, prob=0.0, bow=0.0, parent=None, max_order=None ):
        self.ngram = ngram
        self.prob = prob
        self.bow  = bow
        self.parent = parent
        self.max_order  = max_order
        self.children = {}

    def add_child( self, ngram, prob=0.0, bow=0.0 ):
        """
          Create a new childe node, an instance of LMTree, and
           add it to the hash of children for this node.
        """
        if len(ngram)==1:
            if ngram[0] in self.children:
                return None  #Already have this node
            self.children[ngram[0]] = LMTree( ngram[0], prob, bow, self )
        else:
            n0 = ngram.pop(0)
            if n0 in self.children:
                self.children[n0].add_child( ngram, prob, bow )
            else:
                self.children[n0] = LMTree( n0, 0.0, 0.0, self )
                self.children[n0].add_child( ngram, prob, bow )
        return

    def get_ngram_p( self, ngram, i=0 ):
        """
          Recursive method for retrieving the probability or backoff weight for
           an input ngram. If the ngram exists, the probability is returned.  
           If not, the backoff weight for the highest order ngram prefix of the
           input is returned. Also returns a boolean value indicating whether 
           the first parameter is a prob or bow.
        """
        if i==len(ngram):
            return self.prob, True

        if ngram[i] in self.children:
            return self.children[ngram[i]].get_ngram_p( ngram, i+1 )
        else:
            return self.bow, False


def load_arpa( arpa_file ):
    """
      Load an ARPA format LM into a simple trie structure.
    """
    
    #Initialize the ngram trie
    arpalm = LMTree( "<start>" )
    order = max_order = 0

    for line in open(arpa_file,"r"):
        line = line.strip()
        #Read the model header info and find the max ngram order
        if line.startswith("ngram"):
            max_order = int(re.sub(r"^ngram\s+(\d+)=.*$", r"\1", line))

        #Now insert each ngram into the trie based on its order
        if order>0 and not line.startswith("\\") and not line=="":
            parts = line.split("\t")
            words = parts[1].split(" ")
            if order<max_order:
                if len(parts)==3:
                    arpalm.add_child( words, float(parts[0]), float(parts[-1]) )
                else:
                    arpalm.add_child( words, float(parts[0]), 0.0 )
            else:
                arpalm.add_child( words, float(parts[0]) )
                
        #Extract the current ngram order
        if re.match(r"^\\\d+",line):
            line = re.sub(r"^\\(\d+).*$",r"\1",line)
            order = int(line)

    arpalm.max_order=max_order

    return arpalm

def compute_sentence_prob( arpalm, sentence ):
    """
      Compute the probability of the input sentence.
      Should produce the same output as the SRILM ngram tool,

        $ ngram -lm test.arpa -ppl sent.txt
      or the NGramLibrary ngramapply tool, to within 
      a delta of about 1e-7.  Longer sentences may diverge
      further due to rounding.
    """

    total = 0.0

    #Initialize the ngram stack with the first word
    # in the input sentence
    ngram = [ sentence.pop(0) ]

    #Keep pushing words/tokens onto the stack until
    # there are no more left in the input sentence.
    while len(sentence)>0:
        ngram.append(sentence.pop(0))
        p, is_prob = arpalm.get_ngram_p( ngram )
        total += float(p)

        #If is_prob is false, there is no history 
        # in the model.  This means that 'p' is a backoff
        # weight.  We keep backing-off to lower order
        # ngrams in this case, each time popping the bottom
        # word on our ngram stack.
        while is_prob==False:
            ngram.pop(0)
            p, is_prob = arpalm.get_ngram_p( ngram )
            total += float(p)

    #Return the total log_10 probability
    # of the input sentence
    return total

def retrieve_ngram_prob( arpalm, sentence ):
    """
      Retrieve an individual ngram probability.
      
    """
    total = 0.0

    #If we have a unigram, just return the probability.  We ASSuME that the
    # unigram will be in the vocabulary.
    if len(sentence)==1:
        p, is_prob = arpalm.get_ngram_p( sentence[0] )
        return p
    
    #Find the shortest prefix for which we have some evidence in the model. 
    #This differs from the approach used for the sentence probability
    # where we start with '<s> w1' and proceed to the highest supported 
    # ngram order given the model.
    ngram = [ ]
    pr = 0.0
    while len(ngram)<arpalm.max_order and len(sentence)>0:
        ngram.append(sentence.pop(0))
        pr, is_prob = arpalm.get_ngram_p( ngram )
        if is_prob==False:
            sentence.insert(0,ngram.pop(-1))
            break 

    #Keep pushing words/tokens onto the stack until
    # there are no more left in the input sentence.
    while len(sentence)>0:
        ngram.append(sentence.pop(0))
        p, is_prob = arpalm.get_ngram_p( ngram )
        total += float(p)

        #If is_prob is false, there is no history 
        # in the model.  This means that 'p' is a backoff
        # weight.  We keep backing-off to lower order
        # ngrams in this case, each time popping the bottom
        # word on our ngram stack.
        while is_prob==False:
            ngram.pop(0)
            p, is_prob = arpalm.get_ngram_p( ngram )
            total += float(p)

    if total==0.0:
        total = pr
    #Return the total log_10 probability
    # of the input sentence
    return total


def history_sum( arpalm, ngram_hist ):
    """
      Compute the sum for a given history.
      This will perform a sum over all n+1 grams starting with ngram_hist,
      as well as any backoff probs for items in the vocabulary that do NOT
      occur as an extension of ngram_hist.  

      The result should always be 1.0~ unless the NGram ends in </s>.
    """

    from math import pow

    #The vocabulary is just the set of unigrams
    vocab = set(arpalm.children.keys())
    
    total = 0.0

    #Now compute the sum over all extensions of the history
    for word in vocab:
        ngram = [ w for w in ngram_hist ]
        ngram.append(word)
        total += pow(10,retrieve_ngram_prob( arpalm, ngram ))
    
    return total

def lm_is_normalized( arpalm_file, delta=1e-6 ):
    """
      Determine whether the model is fully normalized.
      
      First, for the zero-th order, compute the unigram sum.
      Next, for each ngram order o<max_order, compute the history_sum.

      If the absolute value,  abs(1-hist_sum)<delta, then count 
      the history as normalized.  Otherwise through a ValueError and quit.
    """
    from math import pow

    order = 0
    arpalm = load_arpa( arpalm_file )
    
    total = 0.0
    #First the unigram model
    for word in arpalm.children.keys():
        total += pow(10., float(arpalm.children[word].prob))
    if abs(1.-total)>delta:
        print total, delta, 1.-total
        raise ValueError, "The unigram model is not fully normalized!"

    total = 0.0
    for line in open(arpalm_file,"r"):
        line = line.strip()
        if order>0 and not line.startswith("\\") and not line=="":
            parts = line.split("\t")
            ngram = parts[1].split(" ")
            total += pow(10., history_sum(arpalm,ngram))
        if re.match(r"^\\\d+",line):
            if order>0 and 1-total > delta:
                raise ValueError, "The %d-order model is not fully normalized!"%(order)
            order = int(re.sub(r"^\\(\d+).*$",r"\1",line))
            #Return True if we've reached the max_order
            if order==arpalm.max_order:
                return True
            #reset the total since we have a new order
            total = 0.0

if __name__=="__main__":
    import sys, argparse

    example = """%s --arpalm lm.arpa --sent "some sentence to evaluate" """ % sys.argv[0]
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument('--arpalm',    "-m",   help="The ARPA format language model to be used.", required=True )
    parser.add_argument('--sent',      "-s",   help="The input sentence/sequence to be evaluated.", required=False, default="" )
    parser.add_argument('--sb',        "-b",   help="The sentence begin token. (<s>)", default="<s>", required=False )
    parser.add_argument('--se',        "-e",   help="The sentence begin token. (</s>)", default="</s>", required=False )
    parser.add_argument('--nosbse',    "-n",   help="Don't add sentence-begin/sentence-end tokens to the input.", default=False, action="store_true" )
    parser.add_argument('--hist_sum',  "-i",   help="Compute the sum for a given history.  History must be in the model.", default=False, action="store_true" ) 
    parser.add_argument('--get_ngram', "-g", help="Retrieve an individual NGram probability. Length of the NGrm  must be <= the max order of the input LM.", 
                        default=False, action="store_true" )
    parser.add_argument('--delta',     "-d", help="Delta for determining normalization success/failure.", type=float, default=1e-6, required=False )
    parser.add_argument('--verbose', "-v", help="Verbose mode.", action="store_true", default=False )
    args = parser.parse_args()

    if args.verbose:
        for attr, value in args.__dict__.iteritems():
            print attr, "=", value

    if args.sent=="":
        print lm_is_normalized( args.arpalm, delta=args.delta )
    else:
        arpalm = load_arpa( args.arpalm )

        tokens = args.sent.split(" ")

        for token in tokens:
            if token not in arpalm.children:
                raise ValueError, "Unigram token: %s not found in LM vocabulary!" % (token)

        if args.get_ngram:
            if len(tokens)>arpalm.max_order:
                print "Sequence:", tokens, "is longer than the max order of the input model!"
                sys.exit(1)
            print "NGram:", tokens
            print "NGram prob:", retrieve_ngram_prob( arpalm, tokens )
        elif args.hist_sum:
            print history_sum( arpalm, args.sent.split(" ") )
        else:
            if args.nosbse:
                print "Not adding sentence-begin/sentence-end tokens."
            else:
                if not tokens[0]==args.sb:
                    tokens.insert(0,args.sb)
                if not tokens[-1]==args.se:
                    tokens.append(args.se)
            print "Evaluating sequence:", " ".join(tokens)
            print "Log_10 prob:", compute_sentence_prob( arpalm, tokens )

    
