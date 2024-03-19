# Shift-reduce parser

## Run

```bash
rm -rf trained_models/models/shift_reduce/toy && allennlp train -s trained_models/models/shift_reduce/toy/ config_files/shift_reduce_toy.jsonnet --include-package minimalist_parser.register_imports 
```
## Dead ends

Dead ends can arise when we try to do crossing dependencies the wrong way.

Suppose we have a sentence abcd and one mover slot.

Merge c and a, and d and b, yielding a stack:

(c, a) | (d, b)

We can't merge the two items (SMC). Adding a silent head to take one of the movers out of storage doesn't work because it's not adjacent -- we don't have any place to put it until the other item is present. There's no way to "tuck in" in this grammar. Even wrap-HM wouldn't work because neither is lexical.

I don't know if there's a way to systematically prevent them. Doesn't seem like it to me, without just filling in the whole parse chart.

### Hacks

#### ListMovers

Mover slots can store a list of movers, take them out by index. This makes the MG Type 0.

#### Don't require adjacency

Let the parser just handle the strings; forget about indices or at least as a last resort. This means the parser is not constrained to build a tree that evaluates to the input sentence.

## Notes for implementation

### Tree LSTM

* Generally, if you implement a data reader and a forward function, AllenNLP will use the various methods of the input Field to connect them, making all the tensors and batching etc.
* However, AllenNLP doesn't have a TreeField, so we need to implement all those methods ourselves.
* Jonas has a TreeField for AM Trees, and I can make my own from his.