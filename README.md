# Minimalist Parsing into minimalist algebra terms

Minimalist Grammars without the features! parse into algebra terms that generate whatever you like.

Public version for MG+2.3

Really, this README is probably not accurate.

## Torr's MGBank corpus

The source of the data


* Split
  * in `data/processed/mg_bank/split`
  * split based on the split used to train the supertagger
  * see `minimalist_parser.convert_mgbank.create_data_split` for details and to do it yourself
* Original corpus in Autobank/MGParse, copied here into data/raw
    * wsj_MGBankSeed
    * wsj_MGBankAuto
* to visualise a tree:
    * inside the MGParse folder:
    * get a tree from one of the corpora above as a string
      * format: JSON
      * keys are numbers as strings, e.g. `"1"`. These are the sentence number from that bit of the PTB
      * values are lists of strings. The first element is the full derivation tree in a format we can visualise as follows.
    * `from gen_derived_tree import gen_derivation_tree, gen_derived_tree, gen_xbar_tree`
    *  `derivation_tree = gen_derivation_tree(tree_as_string)`
    * `derivation_tree.visualize_tree()`

## Generalised Minimalist Algebras

The mathematical foundation of the project

### Major modules:
 * `algebras.algebra` defined algebra terms and algebras
 * `trees.trees` defines trees
 * `minimalism.minimalist_algebra` defines generalised minimalist algebras with main structures as inner algebra terms
 * `algebras.hm_algebra` defined abstract class for inner algebras, for MG that works with the terms (rather than objects) of the inner algebra
 * Some inner algebras:
    * `algebras.hm_triple_algebra.py`    
    * `algebras.hm_interval_pair_algebra`

### More algebras, MG
 * Some more algebras not necessarily yet checked for obsolescence:
    * string_algebra
    * tree_algebra.py makes "bare trees" a la Stabler 1997
    * tuple_algebra.py makes triples of (left of the head, head, right of the head) strings, allowing for head movement, as well as a wrap function like in TAGs
    * tag_algebra.py probably isn't up to date with psi_mga, but it's string pairs for the TAG string algebra
 * `minimalism.minimalist_grammar` implements the feature-driven grammar
   * (incomplete)
   

## Project organization

```
Legend:
PG: Project Generated
RO: Read Only
HW: Human Written

.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── requirements.txt    <- Currently in requirements folder, of dubious accuracy
├── config_files        <- Configuration files (HW)
│                           - yml files for Docker Compose
│                           - jsonnet files for allennlp
├── data                <- All project data, ignored by git
│   ├── processed       <- The final, canonical data sets for modeling. (PG)
│   │   ├── mg_bank     <- mgbank files that are not strictly raw. (PG)
│   │   │   └── split   <- train/dev/test split (official) (PG)
│   │   └── seq2seq     <- train and dev sets for seq2seq models                                
│   ├── raw_data        <- The original, immutable data dump. (RO)
│   └── temp            <- Intermediate data that has been transformed. (PG)
├── docs                <- Documentation notebook for users (HW)
│   ├── manuscript      <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports         <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── results
    ├── analysis        <- analysis results (PG & HW)
│   ├── figures         <- Figures for the manuscript or reports (PG)
│   └── predictions     <- model predictions (PG)
├── scripts             <- bash scripts (HW)
├── tests               <- unit tests (HW)
├── trained_models      <- models and associated files (PG)
└── minimalist_parser   <- Source code for this project (HW)

```