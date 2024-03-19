from ..algebras import algebra
from ..trees.transducer import *
from ..minimalism.movers import ListMovers

# *** Transducers ***

def state_combiner(states):
    """
    combines states with ATB if they're the same and otherwise adding the states together
    @param states: list of states, which can be anything with + and == defined
    @return: new state
    """
    if len(states) == 2 and states[0] == states[1]:
        return states[0]
    new_state = []
    for child in states:
        new_state += child
    return new_state


def dsmc_leaf2list_leaf(qs, leaf):
    """
    changes only the mover type
    @param qs: states to pass on
    @param leaf: AlgebraOp
    @return: states, AlgebraOp pair
    """
    inner_item = leaf.function.inner_term
    return qs, algebra.AlgebraOp(leaf.name, mg.Expression(inner_item, ListMovers(), mg_type=leaf.function.mg_type))


def combine_epp_movers_rule_generator(t: algebra.AlgebraTerm, _, d):
    """
    tree transducer rule generator for combining EPP movers into two slots
    States track the filled mover slots.
    States: dicts of filled new to_slot names : old to_slot names
            only loc, num, pers, and multiple (the "EPP movers") change.
            a new state is just the union of the child states, and then we change the EPP slots as follows:
            if we're adding an EPP mover, put it in E1 if there's room,
             otherwise in E2. Make its value the old name, e.g. E1: loc
            if we're adding a non-EPP mover, just add to_slot: None to the state
                e.g. Abar: None
            if we're removing an EPP mover, rename the function according to its
                new from_slot, which you can find by searching the values of
                E1 and E2 for the right feature
    @param t: input tree
    @param _: was supposed to be the children's states...
    @return: TransitionRule
    """
    if t.children is None:
        # state for leaf is empty dict
        return TransitionRule(dsmc_leaf2list_leaf,
                              lambda trees: type(t)(trees[0]))
    # log(f"parent: {t.parent}")
    if t.parent.mg_op == minimalist_algebra_triples.merge1:
        # merge1 doesn't change anything
        return TransitionRule(lambda qs, parent: (state_combiner(qs), parent),
                              lambda trees: type(t)(trees[0], trees[1:]))

    if t.parent.mg_op == minimalist_algebra_triples.merge2:
        def merge2_state_builder(child_states, parent):
            """
            child states encode their filled mover slots. If we are adding an
                EPP movers, we rename the to_slot with E1, or E2 if E1 is full,
                and track its presence in the value of E1 or E2 accordingly,
            @param child_states: list of 2 dicts from str to str or None
            @param parent: MinimalistFunction
            @return: (new state, new MinimalistFunction)
            """
            new_state = state_combiner(child_states)
            if parent.to_slot.epp:
                new_state.append(parent.to_slot.name)
                new_parent_function = mg.MinimalistFunction(
                    minimalist_algebra_triples.merge2,
                    inner_op=parent.inner_operation,
                    to_slot=slot_name2slot(E),
                    prepare=parent.prepare,
                    adjoin=parent.adjoin
                )
                return new_state, new_parent_function
            else:
                return new_state, parent

        return TransitionRule(merge2_state_builder, lambda trees: type(t)(trees[0], trees[1:]))

    elif t.parent.mg_op == minimalist_algebra_triples.move1:
        def move1_state_builder(child_states, parent: mg.MinimalistFunction):
            new_state = state_combiner(child_states)
            old_slot = parent.from_slot
            log(f"move1 new state: {new_state}, old slot: {old_slot}")
            if old_slot in new_state:  # if it's one of the epp movers
                log("found epp!")
                i = new_state.index(old_slot)
                log(f"index: {i}")
                new_parent_function = mg.MinimalistFunction(
                    minimalist_algebra_triples.move1,
                    inner_op=parent.inner_op,
                    from_slot=E,
                    index=i
                )
                new_state.pop(i)
                log(new_parent_function)
                return new_state, new_parent_function
            else:
                return new_state, parent

        return TransitionRule(move1_state_builder, lambda trees: type(t)(trees[0], trees[1:]))

    elif t.parent.mg_op == minimalist_algebra_triples.move2:
        def move2_state_builder(child_states, parent: mg.MinimalistFunction):
            new_state = state_combiner(child_states)
            # from to_slot
            old_from_slot = parent.from_slot
            if old_from_slot in new_state:
                new_from_slot = E
                i = new_state.index(old_from_slot)
                log(f"\n *** index {i} ***\n")
                new_state.pop(i)  # remove from state
            else:
                new_from_slot = old_from_slot
                i = 0
            # to to_slot
            old_to_slot = parent.to_slot
            if old_to_slot.epp:
                new_to_slot = slot_name2slot(E)
                new_state.append(old_to_slot.name)
            else:
                new_to_slot = old_to_slot
            new_parent_function = mg.MinimalistFunction(
                minimalist_algebra_triples.move2,
                from_slot=new_from_slot,
                to_slot=new_to_slot,
                inner_op=parent.inner_op,
                index=i
            )
            return new_state, new_parent_function

        return TransitionRule(move2_state_builder, lambda trees: type(t)(trees[0], trees[1:]))


# def address2index_rule_generator(t: algebra.AlgebraTerm, states, d):
#     """
#     Transduces a term over a minimalist algebra with inner Triple algebra with
#      address markers on the leaves to one over interval pairs
#     @param t: term over MinimalistAlgebra with inner inner_alg = TripleAlgebra
#     @param states: none, just for compatability
#     @param d: dict from addresses to word indices
#     @return: term over MinimalistAlgebra with inner algebra HMIntervalPairAlgebra
#     """
#     if t.children is None:
#         # states of leaves are based on the node label
#         def make_leaf(in_trees):
#             """
#             extracts address for pronounced leaves, looks them up in d to get
#                 their index i in the sentence,
#                 and makes leaf into an interval (i, i+1)
#             unpronounced leaves yield empty intervals
#             @param in_trees: label::daughters list
#             @return: trivial AlgebraTerm with inner item the new interval
#             """
#             main = in_trees[0].function.inner_term.span
#             coord = in_trees[0].function.type.conj
#             if main.startswith("["):
#                 # silent daughters are all things like [det]
#                 return type(t)(lexical_constant_maker(epsilon_label, pair_alg, DSMCMovers, coord, silent=True))
#             else:
#                 return type(t)(lexical_constant_maker(int(d[main.split(address_sep)[-1]]), pair_alg, DSMCMovers, coord))
#
#         return TransitionRule(lambda _, leaf: (Q, leaf), make_leaf)
#     else:
#         return TransitionRule(lambda qs, parent: (Q, lambda op: mg_op2mg_op(op, pair_alg)),
#                               lambda trees: type(t)(trees[0], trees[1:]))
#

# ***

# def address_leaves(term: algebra.AlgebraTerm):
#     """
#     Appends the addresses of leaves to their labels, separated by address_sep
#     e.g. catADDR_SEP0010
#     """
#
#     def _mark_address(t: algebra.AlgebraTerm, addr: str):
#         """
#         Recursively goes through tree, building addresses and relabeling the
#          leaves of new_tree to include the address
#         @param t: : the subtree we're at right now
#         @param addr: string of 0's and 1's
#         @return: AlgebraTerm
#         """
#         if t.children is None:
#             # old_label = t.parent.name
#             # new_label = f"{old_label}{address_sep}{addr}"
#             new_t = algebra.AlgebraTerm(lexical_constant_maker(new_label, triple_alg,
#                                                                mover_type=DSMCAddressedMovers,
#                                                                # type(t.parent.function.movers),
#                                                                conj=t.parent.function.mg_type.conj,
#                                                                silent=label_is_silent(old_label)
#                                                                ))
#             # len(t.parent.function.inner_term.head) == 0))
#             return new_t
#
#         else:
#             return type(t)(t.parent, [_mark_address(kid, addr + str(i)) for i, kid in enumerate(t.children)])
#
#     # run through the tree and relabel the new one
#     return _mark_address(term, '')
#

def nltk_trees2list_term(plain, annotated, f, i):
    """
    Wrapper function from the two nltk trees read in by
     mgbank_input_codec.read_corpus_file to an AlgebraTerm over
        MinimalistFunctions with inner algebra HMIntervalPairsAlgebra
        and mover store ListMovers
    @param plain: nltk.Tree with internal nodes labelled with MGBank operations
    @param annotated: nltk.Tree with nodes labelled with partial results
    @param f: file name for printing errors
    @param i: sentence number for printing errors
    @return: AlgebraTerm over MinimalistFunctions with inner algebra
                HMIntervalPairsAlgebra and mover store ListMovers
    """

    # currently nltk.Trees
    if VERBOSE:
        log("\n**Annotated\n")
        annotated.draw()
        # log(annotated)
        log("\n** Plain")
        # log(plain)

    # move over to AlgebraTerms and add addresses to node labels
    log("** Build algebra term")
    t = nltk_trees2algebra_term(plain, annotated, mg)

    t = add_addresses(t, triple_alg, address_alg)
    log(f"\n *** algebra term with addresses:\n{t}")

    # get the string output
    s = t.spellout(triple_alg)
    log(f"\nSpellout: {s}\n")

    # # build a dict of the addresses of pronounced leaves and their indices in
    # # the output sentence as built by the algebra term
    # d = {}
    # sentence = []
    # for i, w in enumerate(s.split()):
    #     sentence.append(w.split(address_sep)[0])
    #     for address in w.split(address_sep)[1:]:
    #         d[address] = i
    # log("\n** d\n")
    # log(str(d))

    # check that the output of the term is the same as the original sentence
    # clean_sentence = [word for word in sentence if not word.startswith('[')]
    # clean_sentence = []
    # for word in sentence:
    #     if not word.startswith("["):  # these are silent, e.g. [past]
    #         clean_sentence.append(word)

    original_sentence_list = clean_original_sentence(annotated.label())

    if original_sentence_list != s.split():
        log(f"File {f} sentence {i} discrepancy")
        log(original_sentence_list)
        log(s)

    # get the addresses as a sentence
    adds = t.spellout(address_alg)

    # add intervals
    t = add_intervals(t, adds, interval_algebra)

    # log("\n** transduce to interval term\n")
    # # transduce to an algebra term over interval pairs
    # _, new_t = transduce(t, address2index_rule_generator, d)

    # log(f"\n *** interval term:\n{t}")

    log("\n** interpret interval term\n")
    # interpret the interval pair term
    expr = t.interp(interval_algebra)
    complete = expr.is_complete()

    # return true iff we got a complete expression
    log(f"complete: {complete}")
    if not complete:
        log(f"{expr}")

    # # transduce to term with ListMovers
    # log("\n** transduce to list movers term\n")
    # _, new_t = transduce(new_t, combine_epp_movers_rule_generator)
    #
    # log(f"\nlist movers term:\n{new_t}")

    # # check it
    # log("\n** interpret list term\n")
    # # interpret the interval pair term
    # expr = new_t.evaluate()
    # complete = expr.is_complete()
    #
    # if complete:
    #     return new_t
    # else:
    #     if not complete:
    #         log("not complete")
    #         log(f"{expr}")
    #     return False

def check_parse(plain, annotated, f="", i=0):
    """
    starting with the plain and annotated NLTK trees, get the algebra terms over
     intervals and return True if we in fact have a complete parse item,
    i.e. the inner item collapses into a single Interval and there are no movers
    @param i: index in file of tree (for error printing)
    @param f: name of file tree came from (for error printing)
    @param plain: nltk.Tree with inner nodes labelled with operations
    @param annotated: nltk.Tree with inner nodes labelled with partial results
    @return: bool
    """
    term = nltk_trees2list_term(plain, annotated, f, i)
    # return true iff we got a complete expression
    if term:
        return True
    else:
        return False


def nltk_trees2addressed_list_term(plain, annotated, f="", i=0):
    """
    Wrapper function from the two nltk trees read in by
     mgbank_input_codec.read_corpus_file to an AlgebraTerm over
        MinimalistFunctions with inner algebra HMStringTripleAlgebra
        and mover store ListMovers
    @param i: index in file of tree (for error printing)
    @param f: name of file tree came from (for error printing)
    @param plain: nltk.Tree with internal nodes labelled with MGBank operations
    @param annotated: nltk.Tree with nodes labelled with partial results
    @return: A pair of an AlgebraTerm over MinimalistFunctions with inner algebra
                HMStringTripleAlgebra and mover store ListMovers and
                the sentence as a list of (word, [address list]) pairs
    """

    # currently nltk.Trees
    log("\n**Annotated\n")
    log(annotated)
    log("\n** Plain")
    log(plain)

    # move over to AlgebraTerms and add addresses to node labels
    log("** Build algebra term")
    t = nltk_trees2algebra_term(plain, annotated)

    log(f"Addressing term")

    t = address_leaves(t)

    log(f"\n *** algebra term with addresses:\n{t}")

    # get the string output
    sent = t.evaluate().spellout()

    log(f"\nSpellout: {sent}\n")

    # get sentence as list of (word, addresses) pairs
    sentence = []
    for i, w in enumerate(sent.split()):
        w_and_addresses = w.split(address_sep)
        sentence.append((w_and_addresses[0], w_and_addresses[1:]))
    log("\n** sentence and addresses\n")
    log(str(sentence))

    # check that the output of the term is the same as the original sentence
    original_sentence_list = clean_original_sentence(annotated.label())

    if original_sentence_list != [w for w, _ in sentence]:
        log(f"File {f} sentence {i} discrepancy")
        log(original_sentence_list)
        log([w for w, _ in sentence])

    # transduce to term with ListMovers
    log("\n** transduce to list movers term\n")
    _, new_t = transduce(t, combine_epp_movers_rule_generator)

    log(f"\nlist movers term:\n{new_t}")

    return new_t, sentence


def nltk_trees2string_list_term(plain, annotated, f=None, i=None, errors_ok=False):
    """
    Wrapper function from the two nltk trees read in by
     mgbank_input_codec.read_corpus_file to an AlgebraTerm over
        MinimalistFunctions with inner algebra HMTripleAlgebra
        and mover store ListMovers,
    @param errors_ok: if true, return a term and sentence if possible. If false, raise any errors that arise.
                OK errors are in interpreting the tree; unavoidable errors are in transforming it.
    @param i: int: this is the ith sentence in the file, for error printing.
    @param f: the file we're working on, for error printing.
    @param plain: nltk.Tree with internal nodes labelled with MGBank operations.
    @param annotated: nltk.Tree with nodes labelled with partial results.
    @return: A pair of an AlgebraTerm over MinimalistFunctions with inner algebra
                HMStringTripleAlgebra and mover store ListMovers and the sentence as a string.
    """

    if VERBOSE:
        # currently nltk.Trees
        log("\n**Annotated\n")
        # log(annotated)
        annotated.draw()

        log("\n** Plain")
        # log(plain)
        plain.draw()

    #    log(trees.Tree.nltk_tree2tree(plain).latex_forest())
    #    log(trees.Tree.nltk_tree2tree(annotated).latex_forest())

    # move over to AlgebraTerms and add addresses to node labels
    log("** Build algebra term")
    t = nltk_trees2algebra_term(plain, annotated)

    try:
        # log(f"\n *** algebra term:\n{t}")
        # log(t.latex_forest())
        if VERBOSE:
            t.function_tree().to_nltk_tree().draw()
    except Exception as e:
        print(f"\nFile {f} sentence {i}")
        print("warning: error transforming algebra term to nltk tree and drawing it:", e)
        if not errors_ok:
            raise e

    # spell out
    try:
        expression = t.interp()
        if VERBOSE:
            expression.inner_term.to_nltk_tree().draw()
        sent = expression.spellout()
        log(f"spellout of term: {sent}")
    except Exception as e:
        print(f"\nFile {f} sentence {i}")
        print("warning: error in spellout of term", e)
        if not errors_ok:
            raise e

    # transduce to term with ListMovers
    log("\n** transduce to list movers term\n")
    _, new_t = transduce(t, combine_epp_movers_rule_generator)

    original_sentence_list = clean_original_sentence(annotated.label())

    try:
        log(f"\nlist movers term:\n{new_t}")

        sent = new_t.evaluate().spellout()
        log(f"\nSpellout: {sent}\n")

        # check that the output of the term is the same as the original sentence
        if " ".join(original_sentence_list) != sent:
            print(f"discrepancy in file {f} sentence {i}")
            print("original:   ", " ".join(original_sentence_list))
            print("transformed:", sent)

    except Exception as e:
        print(f"\nFile {f} sentence {i}")
        print("warning: error in printing or spelling out list movers term:", e)
        if not errors_ok:
            raise e
        sent = " ".join(original_sentence_list)

    return new_t, sent