# Analysing predictions

`minimalist_parser.analysis.evaluate_predictions` reads in predictions and gold terms and reports a bunch of information, and creates Vulcan-readable pickles for hand-analysis.

Because of the way Python (mis-)handles packages and scripts, you have a couple of options, none of which is ideal:

1. import `minimalist_parser.analysis.evaluate_predictions.main` into a top-level python file such as `sandbox.py` and give it the paths from the top level to the files
2. From the command line, from the top level, you need to include  `PYTHONPATH=./` 
   ```bash
   PYTHONPATH=./ python minimalist_parser/analysis/evaluate_predictions.py data/processed/seq2seq/sampled/to_predict.tsv results/predictions/bart/predictions.txt results/analysis/bart/
   ```
3. Run `evaluate_predictions.py` in PyCharm. I don't know why this works. Might work in Spyder too?

## Outputs and what to do with them

### Evaluation results 
The script in `evaluate_predictions.py` writes basic output to the console and full output to a logfile in `logfiles/analysis/bart/predictions.log`. This includes which trees are interpretable, what errors are thrown by which trees, which trees evauate to the wrong sentence (and what the sentences are), which trees are an exact match to the gold trees, and precision, recall, and F1 for various improvised tree-similarity measures, and possibly more.

### Trees to look at by hand

If you give the script a `pickle_path` it will also generate pickles of various sets of trees. These are designed to be read by the visualisation tool Vulcan.

Repo:
https://github.com/jgroschwitz/vulcan

To use it, first Git-clone it to your computer, and then follow the instructions on the README. It'll open up in your browser (if not, copy the URL it gives you and paste it into your browser.)

```bash
PYTHONPATH=./ python vulcan/launch_vulcan.py path/to/pickle-file
```

e.g.

```bash
PYTHONPATH=./ python vulcan/launch_vulcan.py ../minimalist_parser/results/analysis/bart/official_split/all.pickle
```

Different pickles have different trees in them. Some have two Minimalist Algebra terms for comparison, and some have the Minimalist Algebra term and the inner term.

#### Getting different pickles

This is actually pretty easy. In `analysis/to_vulcan.py` you can find the functions I'm currently using, and just copy, paste, and modify them, then call them or add them to `evaluate_predictions.compare_predictions_to_gold` or whatever you like.

The crucial thing is to make sure that the names of the trees/strings that you give when you initialise the `PickleBuilder` are the same as the ones you add trees/strings to.