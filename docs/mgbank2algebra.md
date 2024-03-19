# MGBank

We need to get training data from the MGBank corpus files. This document explains how that works.

Note much of this might be out of date.

## Quickstart

If they exist, the files containing the terms we want for training should be in `output_files`, with the terms from the seed in subfolder `seeds/` and the auto-generated ones in `auto/`. 

To make them, you can run `mgbank2training_data.py` with the first argument path to the MGBank parent folder (e.g. `MGbankAuto`) and the second argument where you want to put the output terms.

Note that some corpus trees violate the DSMC, so we expect around 3% to raise an error

```bash
PYTHONPATH=./ python mgbank2training_data.py path/to/mgbank output/directory/path out_file_name_without_suffix
```

Only trees without errors are written to file.

### major modules
   * `hm_pair_algebra_mgbank` is a head movement pair algebra optimised for MGBank
   * `hm_triple_algebra_mgbank_strings` is a head movement string triple algebra optimised for MGBank

   * `mgbank_parser` implements the parse items and transition rules for an MGBank-based parser
   * `mgbank2algebra` converts trees from the MGBank tree bank to terms over a minimalist algebra with Interval pairs
   * `movers` implements the mover storage functionality
   * `mgbank2training_data` mostly wrapper functions for creating training data

### MGBank operations vs my operations

The goal is, for every case distinction in MGBank, to provide a function that implements the string-manipulation part
of the effect. Some operations should multiply by number of mover slots, since MGBank uses lists of movers and their
features. Others merge because only the features are treated differently, or the case distinction is taken care of,
e.g. whether the movers come from the functor or the argument.

### Notation etc

 * the objects of both systems are called expressions. For MGBank, expressions are tuples of:
   * string triple (leftward material, head, rightward material)
   * type (+/-lexical, +/-coordination, etc)
   * feature stack
   * mover list of expressions, or maybe just strings paired with features


 
### Head Movement with triples vs pairs

The inner item in MGBank is a triple, as is traditional, but Milos and I simultaneously noticed that you can actually
do it with typed pairs. The reason is that a triple of intervals that has two gaps in it is a dead end: nothing applies
to it. So instead we have just a pair. Since MGBank has no affix lowering, we can keep it pretty simple: (head, rest).
The type, aside from lexical and conjunction, encodes whether or not you MUST apply HM when you select this item. 

Milos just split every rule into must/may not, but I want to prevent making decisions before they're
necessary, so whether you must HM depends on:
   * A merge that creates a gap. both leftward and rightward material are supposed to concatenate with the head, and if they can't, it must be because the head moves
   * A leftward merge of an interval that's actually to the right of the head, or vice versa. Since this depends on the derivational history, we have to carry the info forward. It's this second case that forces the typing

I also allow Intervals, not just Pairs, as Items. You can't HM exactly when you have a non-lexical Interval, because they get created when you merge material to both the left and right of the head.

### Mover storage

* class `Movers`: contains a dict for storing movers, and controls SMC
* subclasses of `Movers`:
    * `DSMCMovers`: implements the complicated DSMC stuff with A movers
        blocking "EPP" movers, which otherwise may co-occur with up to 2
    * `DSMCAddressedMovers`: like `DSMCMovers` but words have the form word.address,
        where address is a sequence of 0's and 1's. For ATB purposes,
        word1.add1 == word2.add2 iff word1 == word2 and the new word
        is word1.add1.add2
      * Addresses are needed to get interval variant, because you can't tell until you build and interpret the 
        term where exactly the various words came from in the MG term.
    * `ListMovers`: WARNING: this makes the MG type-0 because it includes Slots with
        the property multiple, where a list of movers is allowed.
        Currently does not implement the DSMC A/EPP stuff but it could.
        In a multiple slot, movers are added to the end, but can be
        removed by index


### MGBank operations that affect only the features

These I just delete. Could maybe keep smove and move_ctrl as move functions?

* fcide
* smove
* move_ctrl
* move2-dot
* type-saturation

### Issues:

#### Coordination

coordinators allow some special stuff:

* =x feature can persist so that you can get multiple conjuncts
* only with coord phrases can you have excorporation, atb hm, and atb phrase movement
* marked with a type, notated with a bar on the separator. I give it to the inner item itself to pass along the licensing  requirements to the place where atb and excorp actually happen.


#### Covert vs Overt movement

MGBank treats movement of pronounced material very differently from movement of silent material. 

1. Movement of silent material is both covert move and movement of things that happen to be silent
2. Regular SMC applies to silent movers: no more than one mover per feature
3. Derelativised SMC (DSMC) applies to pronounced movers: no more than one per slot

I skip all such movement. Anytime MGBank stores something silent, I don't store it at all.

#### Rightward movement

Rightward mover has a normal sel feature as its category, but also has a sel~ feature in its licensee features, meaning
that the next time a phrase of that category is complete (i.e. its category matches the sel~ feature), rightward movement applies.

e.g. verb : =d v  +  noun : d c~ -> verb : v, noun:c~
does :: v= c + verb : v, noun:c~ -> does verb : c, noun:c~ -> does verb noun : c

So for merge2 and move2 (from=R), we can see by the mover that it gets stored in R.
For move (to=R) we see not by a +f but rather just an f on the head.  

#### Control

D feature on DP (as opposed to d) can optionally persist and then be checked by Move with =d licensor.

merge_ctrl1/2/3 are merge2-A ops.

He doesn't list the move1 version (which is different from a normal move1 op only in the features).

#### Identity operations

Diacritics ! and ? can appear at the end of a licensor feature.
Used to license various sorts of ld agreement where there might not actually be anything there to agree with.

! deleting suicidal licensor e.g. +f!
? non-deleting


### Obsolete, but I think these are all Torr's functions
#### Merge 1: merge a non-mover

Note this is a combination of traditional Merge 1 and 2
These functions don't exist anymore, but maybe there's something useful I can update to

   * `mg_right` encodes 
     * merge1
     * adjoin2
     * h_coord1
   * `mg_left` encodes
     * merge2
     * merge3
     * adjoin1
     * merge3 in app B.10
     * h_coord2
     * h_coord3
     * merge_esc_1
     * merge_atb1
   * `mg_hm_suf` encodes
     * merge_hm3
   * `mg_hm_pre` encodes
     * merge_hm1
   * `mg_excorp` encodes
     * mrg_excorp
   * `mg_atb` encodes
     * merge_atb3?? look up

#### Merge 2: merge a mover

Note this is traditionally called Merge 3. mg_SLOT stores the mover in SLOT.

   * `mg_A` encodes, for A-movers,
     * merge4
     * merge5
     * merge6
     * merge_esc_2
     * merge with ATB phrase movement
   * `mg_Abar` encodes, for A'-movers,
     * merge4
     * merge5
     * merge6
     * merge_esc_2
     * merge with ATB phrase movement
   * `mg_C` encodes
     * merge_ctrl1
     * merge_ctrl2
     * merge_ctrl3
   * `mg_Abar_hm_suf` encodes, for A'-movers,
     * merge_hm_4
   * `mg_Abar_hm_pre` encodes, for A'-movers,
     * merge_hm_2
   * `mg_A_hm_suf` encodes, for A-movers,
     * merge_hm_4
   * `mg_A_hm_pre` encodes, for A-movers,
     * merge_hm_2
   * `mg_A_covert_right` encodes, for A-movers,
     * p_merge1
     * p_adjoin2
   * same for Abar and C
   * `mg_A_covert_left` encodes, for A-movers,
     * p_merge2
     * p_merge3
     * p_adjoin1
   * same for Abar and C

#### Move 1: final move

* `mv_A`, `mv_Abar`, `mv_C` encode, for A/A'/C-movers respectively,
  * move1
* `mv_A_right`, `mv_Abar_right`, `mv_C_right` encode, for A/A'/C-movers respectively,
  * r_move

#### Move 2: non-final move

Currently only encode a small subset of possible slot pairs because I don't think John allows improper movement

* `mv_A_Abar, mv_Abar_Abar, mv_C_Abar` encode
  * move2

