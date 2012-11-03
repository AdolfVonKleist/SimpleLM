#!/usr/bin/python
from math import log, pow, e
import re

class ARPA2WFSA( ):
    """
      Transform a statistical language model in ARPA format
      to an equivalent Weighted Finite-State Acceptor.
      This implementation adopts the Google format for the output
      WFSA.  This differs from previous implementations in several ways:

       Start-state and <s> arcs:
         * There are no explicit sentence-begin (<s>) arcs
         * There is a single <s> start-state.

       Final-state and </s> arcs:
         * There are no explicit sentence-end (</s>) arcs
         * There is no explicit </s> state
         * NGrams ending in </s> are designated as final
            states, and any probability is assigned 
            to the final weight of said state.
    """

    def __init__( self, arpa_file, prefix="test", sb="<s>", se="</s>", eps="<eps>", phi="<phi>", usefail=False, logE=True ):
        self.arpa_file = arpa_file
        self.sb        = sb
        self.se        = se
        self.eps       = eps
        self.phi       = phi
        self.usefail   = False
        self.prefix    = prefix
        #If max_order==0 we'll rely on the ARPA model header
        self.max_order = 0
        self.order     = 0
        #LMs are typically encoded using log base 10
        # But OpenFst and AT&T FSMtools use log base e
        #We convert by default.
        self.logE      = logE
        self.isyms     = set([])
        self.ssyms     = set([])

    def _log10_to_logE( self, val ):
        """
          Convert 'val' from log-base 10 to
          log-base e.  
        """
        return -1 * log( pow(10.,val), e )

    def _make_final( self, s_st, weight ):
        """
          Create a final state.
        """

        if self.logE==True:
            weight = self._log10_to_logE( weight )
        
        self.ssyms.add(s_st)

        print "%s\t%0.7f" % (s_st, weight)

        return

    def _make_arc( self, s_st, e_st, isym, weight ):
        """
          Generate an arc from the required inputs.
        """

        if self.logE==True:
            weight = self._log10_to_logE( weight )

        self.isyms.add(isym)
        self.ssyms.add(s_st)
        self.ssyms.add(e_st)

        print "%s\t%s\t%s\t%0.7f" % (s_st, e_st, isym, weight)

        return

    def arpa_to_wfsa( self ):
        """
          Convert a text-based ARPA format Language Model to 
          WFSA format.  This tool utilizes the Google WFSA format
          where neither the sentence-begin (<s>) nor sentence-end
          (</s>) tokens are explicitly represented.

          The model is expected to be in the following standardized format:

              \data\
              ngram 1=M
              ngram 2=M
              ...
              ngram N=M

             \1-grams:
             p(w)      w     bow(w)
             ...
             \2-grams:
             p(v,w)    v w   bow(v,w)
             ...
             \3-grams:
             p(u,v,w)  u v w

             \end\

          where M refers to the number of unique NGrams for this order,
          and N refers to the maximum NGram order of the model.  
          Similarly, p(w) refers to the probability of NGram 'w', and
          bow(w) refers to the back-off weight for NGram 'w'.  The highest
          order of the model does not have back-off weights.  Back-off
          weights equal to 0.0 in log-base 10 may be omitted to save space,
          and NGrams ending in sentence-end (</s>) naturally do not have 
          back-off weights.

          The NGram columns are separated by a single tab (\t).
          
        """
        
        for line in open(self.arpa_file,"r"):
            line = line.strip()
            
            if self.order>0 and not line.startswith("\\") and not line=="":
                ngram = re.split(r"\s+", line)
                prob  = float(ngram.pop(0))
                bow = 0.0
                if len(ngram)>self.order: bow = float(ngram.pop(-1))

                #We have a unigram model - just requires a single state
                if self.max_order==1:
                    if ngram==self.sb or ngram==self.se:
                        #Skip sentence-begin, sentence-end in the 1-gram case
                        continue
                    self._make_arc( self.sb, self.sb, ngram[0], prob )
                    self._make_final( self.sb, 1.0 )

                elif self.order==1:
                    if ngram[0]==self.sb:
                        #Just a back-off weight
                        self._make_arc( self.sb, self.eps, self.eps, bow )
                    elif ngram[0]==self.se:
                        #Just a probability
                        self._make_final( self.eps, prob )
                    else:
                        self._make_arc( self.eps, ngram[0], ngram[0], prob )
                        self._make_arc( ngram[0], self.eps, self.eps, bow )

                elif self.order<self.max_order:
                    isym  = ngram[-1]
                    s_st = ",".join(ngram[:-1])
                    if isym==self.se:
                        self._make_final( s_st, prob )
                    else:
                        e_st = ",".join(ngram)
                        b_st = ",".join(ngram[1:])
                        self._make_arc( s_st, e_st, isym, prob )
                        self._make_arc( e_st, b_st, self.eps, bow )
                elif self.order==self.max_order:
                    isym = ngram[-1]
                    s_st = ",".join(ngram[:-1])
                    if isym==self.se:
                        self._make_final( s_st, prob )
                    else:
                        e_st = ",".join(ngram[1:])
                        self._make_arc( s_st, e_st, isym, prob )
                    

            elif line.startswith("ngram"):
                self.max_order = int(re.sub(r"^ngram\s+(\d+)=.*$", r"\1", line))

            elif re.match(r"^\\\d+",line):
                self.order = int(re.sub(r"^\\(\d+).*$", r"\1", line))

        return

    def print_syms( self ):
        if self.eps in self.isyms: self.isyms.remove(self.eps)
        if self.phi in self.isyms: self.isyms.remove(self.phi)
        if self.eps in self.ssyms: self.ssyms.remove(self.eps)

        self._print_isyms( )
        self._print_ssyms( )
        
        return 

    def _print_isyms( self ):
        ofile_n = "PREFIX.isyms".replace("PREFIX",self.prefix)
        ofp     = open(ofile_n, "w")
        ofp.write("%s 0\n" % (self.eps))
        ofp.write("%s 1\n" % (self.phi))
        for i,sym in enumerate(self.isyms):
            ofp.write("%s %d\n" % (sym, i+2))
        return

    def _print_ssyms( self ):
        ofile_n = "PREFIX.ssyms".replace("PREFIX",self.prefix)
        ofp     = open(ofile_n, "w")
        ofp.write("%s 0\n" % (self.eps))
        for i,sym in enumerate(self.ssyms):
            ofp.write("%s %d\n" % (sym, i+1))
        return

if __name__=="__main__":
    import sys

    converter = ARPA2WFSA( sys.argv[1] )
    converter.arpa_to_wfsa( )
    converter.print_syms( )
