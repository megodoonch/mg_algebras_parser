# Minimalist Parsing into minimalist algebra terms

Minimalist Grammars without the features! parse into algebra terms that generate whatever you like.


**WARNING**: 

* recently reorganised files. There are probably some problems with imports still. 
* Python scripts need to be run with `PYTHONPATH=./` preposed to them. Not tested yet with allennlp

## Set up

###  Quick start with pip:

I don't know if this makes sense anymore, but I'm making a conda environment and then installing everything with pip

```bash
conda create -n env_name python=3.9.12
conda activate env_name
pip install -r requirements.txt
```

### Starting over with pip

This is specifically with cuda toolkit version 11.3, as the new GPUs on Surfsara have `sm_86` something (architechture?) and 10 isn't compatable.
Conda doesn't seem to allow me to install cuda toolkit on my local machine, but pip does.

Check cuda version: this should get 11.3. If you get None, I think that means your torch doesn't have cuda toolkit even if you asked for it.

```python
import torch
print(torch.version.cuda)
```

Make a new conda environment for some reason, and specify the python version.
pip install pytorch instructions here: https://pytorch.org/get-started/locally/

```bash
conda create -n new_env_name python=3.10
conda activate new_env_name
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install allennlp
pip install allennlp-models
```

If you want a docker image with exactly these packages: 

1. pipe pip freeze to requirements.txt 
2. remove any weird @file.... things
3. remove
    ```
    torch==1.12.1+cu113
    torchaudio==0.12.1+cu113
    torchvision==0.13.1+cu113
    ```
4. and make a new docker image



### More Docker

To tag the image with multiple tags or give them multiple names use multiple `-t <name>`s

and to give the container your own name, e.g. `mg-parser-container`: 

```bash
docker build --progress=plain -t mg-parser:latest -t mg-parser:0.x . --no-cache
docker run -it --name mg-parser-container mg-parser
```

To retag or rename:

```bash
docker tag oldname:oldtag newname:newtag
```

In particular, to rename so that it can be pushed to dockerhub:

```bash
docker tag mg-parser:my_tag megodoonch/mg-parser:my_tag
docker push megodoonch/mg-parser:my_tag
```

### Other

To get the requirements from a working environment:

```bash
pip list --format=freeze > requirements.txt
```

To pipe all output to a logfile and also keep it in the terminal, put this at the end of the command:

```bash
 2>&1 | tee path_to_logfile
```

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
 * `possibly_obsolete_algebras` contains old versions of hm algebras and minimalist algebras
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