"""
visualisation script for MGBank files, using NLTK
run python visualise_mg_bank.py -h for instructions
"""

import argparse

if __name__ == "__main__":

    # default filepath
    filepath = "../../data/raw_data/MGBank/wsj_MGbankSeed/00/wsj_0001.mrg"

    parser = argparse.ArgumentParser(description="Choose tree types to display from file. "
                                                 "They will be displayed in a pop-up window, "
                                                 "which you must close to see the next tree")
    parser.add_argument('-f', '--file', help="path to the corpus file, ending in .mrg", default=filepath)
    parser.add_argument('-x', '--xbar', action='append_const', const=1, dest='trees', help='display the Xbar tree')
    parser.add_argument('-d', '--derivation', action='append_const', const=4, dest='trees',
                        help='display the derivation tree without the unification/agreement features')
    parser.add_argument('-a', '--annotated', action='append_const', const=5, dest='trees',
                        help='display the derivation tree, annotated with partial results,'
                             ' without the unification/agreement features')
    parser.add_argument('-df', '--derivation_full', action='append_const', const=0, dest='trees',
                        help='display the full derivation tree')
    parser.add_argument('-b', '--bare', action='append_const', const=2, dest='trees',
                        help='display the bare tree')
    parser.add_argument('-af', '--annotated_full', action='append_const', const=3, dest='trees',
                        help='display the derivation tree annotated with partial results')
    parser.add_argument('-n', '--sentence_number', default=0, type=int,
                        help='when there is more than one sentence in the file, display the one at the given index')

    # The trees in the JSON file:
    # full_derivation = 0
    # xbar = 1
    # with_arrows = 2
    # annotated = 3  # internal nodes labeled with intermediate results (full features)
    # derivation_no_unification_features = 4
    # annotated_no_unification_features = 5

    tree_indices = {0: "full derivation tree with unification features",
                    1: "Xbar tree",
                    2: "bare tree",
                    3: "annotated derivation tree with unification features",
                    4: "derivation tree",
                    5: "annotated derivation tree"
                    }

    args = parser.parse_args()
    print(f"getting tree {args.sentence_number} from {args.file}")
    if args.trees is None:
        # display the full derivation tree without unification features by default
        args.trees = [4]
    try:
        for tree in args.trees:
            print("displaying", tree_indices[tree])
            t = mgbank_corpus_file2nltk_trees(args.file, tree)[args.sentence_number]
            t.draw()
    except IndexError:
        print(f"No tree at index {args.sentence_number} in file {args.file}")
