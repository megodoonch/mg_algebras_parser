class MGType:
    """
    Implements the type of an expression

    Attributes:
        lexical: boolean: False if the item has already had an operation
                            applied
        conj: boolean: true if a conjunction or we're busy building a
                        coordinated phrase
    """

    def __init__(self, lexical=True, conj=False):
        """
        Creates an instance of a type for minimalist grammars
        @param lexical: False for derived items
        @param conj: True for conjunctions and Coord'/CoordP phrases.
         Controls ATB and excorporation.
        """
        self.lexical = lexical
        self.conj = conj

    def __repr__(self):
        string = ":"
        if self.lexical:
            string += ":"
        if self.conj:
            string += "c"
        return string

    def __eq__(self, other):
        return self.conj == other.conj and self.lexical == other.lexical

    @staticmethod
    def string2item_and_type(string: str):
        """
        given a string of the representation of an item followed by its type
        , extracts the string of the item and the type
        @param string: str
        @return: str, MGType
        """
        lex = False
        conj = False
        if "::" in string:
            lex = True
        parts = string.split(":")
        item = parts[0]
        if len(parts) == 1:
            return item, MGType(lexical=lex)
        if len(parts) > 1:
            if "c" in parts[-1]:
                conj = True
        return item, MGType(lexical=lex, conj=conj)
