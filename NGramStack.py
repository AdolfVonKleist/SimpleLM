#!/usr/bin/python


class NGramStack( ):
    """
      A stack object designed to pop and push word 
      tokens onto an N-gram stack of fixed max-length.
    """
    
    def __init__( self, order=3 ):
        #Maximum span for N-grams in this model.
        self.o = order 
        #The actual stack
        self.s     = []

    def push( self, word ):
        """ 
           Push a word onto the stack.
           Pop off the bottom word if the
           stack size becomes too large.
        """
        self.s.append(word)
        if len(self.s)>self.o:
            self.s.pop(0)
        return self.s[:]

    def pop( self ):
        self.s.pop(0)
        return self.s[:]

    def clear( self ):
        """
           Empty all the words from the stack.
        """
        self.s = []
