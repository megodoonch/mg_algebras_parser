import sys

from mgbank_input_codec import *
import nltk

if __name__ == "__main__":

    # default; can be given from command line
    filepath = "C:/Users/frede/Documents/UU/Taalwetenschap/Jaar 3/Thesis/Autobank/MGParse/wsj_MGbankSeed/00/wsj_0001.mrg"

    # The trees in the JSON file:
    full_derivation = 0
    xbar = 1
    with_arrows = 2
    annotated = 3  # internal nodes labeled with intermediate results (full features)
    derivation_no_unification_features = 4
    annotated_no_unification_features = 5

    # use these values
    sentence_number = 1
    j = 0

    if len(sys.argv) > 1:
        filepath = sys.argv[1]


    t = mgbank_corpus_file2nltk_trees(filepath, j)[sentence_number]
    t.draw()
