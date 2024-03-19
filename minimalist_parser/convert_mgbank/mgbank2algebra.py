"""
Module for converting MGBank trees to mg algebra terms

"""
import os
import pickle
import re
from copy import deepcopy

import dill
import nltk

from minimalist_parser.algebras.algebra_objects.intervals import Interval, PairItem
from minimalist_parser.algebras.hm_algebra import HMAlgebra
from minimalist_parser.minimalism.mg_errors import SMCViolation, MGError
from minimalist_parser.minimalism.prepare_packages.prepare_packages_hm import PreparePackagesHM, ATBError
from ..minimalism.minimalist_algebra_synchronous import MinimalistAlgebraSynchronous, MinimalistFunctionSynchronous, SynchronousTerm, InnerAlgebraInstructions
from ..minimalism.minimalist_algebra import MinimalistAlgebra
from ..algebras import hm_algebra, algebra
from ..minimalism.mg_types import MGType
from ..convert_mgbank.slots import slot_name2slot, A, Abar, R, Self, E
from ..minimalism import minimalist_algebra as mg
from ..trees.transducer import TransitionRule, transduce, Q
from ..minimalism.movers import Slot, ListMovers, DSMCMovers, DSMCAddressedMovers
from ..algebras.hm_triple_algebra import HMTripleAlgebra
from ..algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from ..minimalism.prepare_packages.triple_prepare_package import HMTriplesPreparePackages
from ..minimalism.prepare_packages.interval_prepare_package import IntervalPairPrepare
from ..minimalism.prepare_packages.addressed_triple_prepare_package import HMAddressedTriplesPreparePackages
from ..algebras.algebra_objects.triples import Triple
from ..algebras.algebra import Algebra


# logging
import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


# make log() into a logging function (for simplicity)
VERBOSE = True
log = logger.debug

# VERBOSE = False



# make the algebras and prepare packages
triple_alg = HMTripleAlgebra()
triple_prepare = HMTriplesPreparePackages()
pair_alg = HMIntervalPairsAlgebra(name="Intervaletjes!")
pair_prepare = IntervalPairPrepare()
address_alg = HMTripleAlgebra("addresses algebra", component_type=list, addresses=True)
address_prepare = HMAddressedTriplesPreparePackages()
minimalist_algebra_triples = MinimalistAlgebraSynchronous(inner_algebras=[triple_alg],
                                                          prepare_packages=[triple_prepare],
                                                          mover_type=DSMCMovers)



# def log(s):
#     if VERBOSE:
#         print(s)


# some symbols
over_line = "\u0305"
epsilon = "\u03b5"
epsilon_label = "epsilon"
coord_sep = "̅:"  # be careful, I don't know where this came from or how to type it. Here's a copy: "̅:"
conj_marker = "<+conj>"

# Some dicts for relationships we need
# from MGBank: positive feature polarities

right_merge = re.compile(r'^\w+=[?!]*$')
right_merge_left_h_move = re.compile(r'^>\w+=[?!]*$')
right_merge_right_h_move = re.compile(r'^\w+<=[?!]*$')
right_merge_x_h_move = re.compile(r'^\w+=\^[?!]*$')  # excorp
left_merge = re.compile(r'^=\w+[?!]*$')
left_merge_left_h_move = re.compile(r'^=>\w+[?!]*$')
left_merge_right_h_move = re.compile(r'^=\w+<[?!]*$')
left_merge_x_h_move = re.compile(r'^=\w+\^[?!]*$')  # ATB HM
left_adjoin = re.compile(r'^\w+≈[?!]*$')
right_adjoin = re.compile(r'^≈\w+[?!]*$')
left_move = re.compile(r'^\+\w+[?!]*$')

# dict from MGBank polarities to Triple algebra inner ops
# we need these because we start with the string triple algebra
# given that we're using prepare, we just use concat
polarity2inner_op = {
    right_merge: triple_alg.concat_right,
    right_merge_left_h_move: triple_alg.concat_right,
    right_merge_right_h_move: triple_alg.concat_right,
    right_merge_x_h_move: triple_alg.concat_right,
    left_merge: triple_alg.concat_left,
    left_merge_left_h_move: triple_alg.concat_left,
    left_merge_right_h_move: triple_alg.concat_left,

    # merge new+old to right of ATB-moved head for eg
    # (_, the, cat and dog) with new = cat
    left_merge_x_h_move: triple_alg.concat_right,
    left_adjoin: triple_alg.concat_left,
    right_adjoin: triple_alg.concat_right,
    left_move: triple_alg.concat_left,
}

# dict from MGBank polarities to Triple algebra prepare packages
# Packages of operations that return an updated pair of functor and other
polarity2prepare = {
    right_merge: None,
    right_merge_left_h_move: triple_prepare.prefix,
    right_merge_right_h_move: triple_prepare.suffix,
    right_merge_x_h_move: triple_prepare.excorporation,
    left_merge: None,
    left_merge_left_h_move: triple_prepare.prefix,
    left_merge_right_h_move: triple_prepare.suffix,
    left_merge_x_h_move: triple_prepare.hm_atb,
    left_adjoin: None,
    right_adjoin: None,
    left_move: None,
}

# some special features
epp_move = re.compile(r'(multiple)|(num)|(pers)|(loc)')  # can co-occur
# covert_only_move = re.compile(r"(negs)|(pol)")


# dict from to_slot names to regular expressions that find the features
# that are stored in those slots
slot2feats = {
    A: re.compile(r'^((-case)|(d)|(-tough))$'),  # the main A to_slot
    Abar: re.compile(r'^-((wh)|(foc)|(top))$'),
    R: re.compile(r'^[tvpcdD]~|(-n)$'),
    Self: re.compile(r'-self'),
}

# mover slots for feature names
feat2slot = {
    "case": A,
    "D": A,
    "tough": A,
    "multiple": "multiple",
    "num": "num",
    "pers": "pers",
    "loc": "loc",
    "epp": "epp",
    "wh": Abar,
    "foc": Abar,
    "top": Abar,
    "t": R, "v": R, "c": R, "d": R, "n": R,
    "self": Self,
    "negs": None, "pol": None
}


# some info about functions can be found in the original MGBank function labels
def is_adjoin(op: str):
    """
    True iff "adjoin" appears in MGBank function name
    @param op: str
    @return: bool
    """
    return "adjoin" in op


def functor_right(op: str):
    """
    MGBank has daughters in string order,
        so right child of left merge is functor.
    functor is in adjoin is adjunct. Note that the head is not the functor
    @param op: MGBank function name
    @return: bool: True if the functor is on the right instead of the left
    """
    return "r_adjoin" in op or "l_merge" in op


def head_right(op: str):
    """
    MGBank has daughters in string order, so the right daughter of left_merge or
     left_adjoin is the head
    @param op: MGBank function name
    @return: True if the head is on the left instead of the right
    """
    return "l_adjoin" in op or "l_merge" in op


def is_ident_function(arity: int, op: str):
    """
    All unary functions that don't have move in their name use identity function
     in the algebra -- they just change the features
    @param arity: int: the number of daughters of the op
    @param op: MGBank operation name
    @return: bool: True if algebra op is the identity function
    """
    return arity == 1 and "move" not in op


# *** Utility Functions ***

def split_item(item: str):
    """
    coordination separators print really weirdly and this seems to take care of all the possibilities
    @param item: str of the item in MGBank format
    @return: inner item string rep, lexical, coordinator, separator string, features as list
    """
    # log(f"splitting item {item}")
    lexical = False
    coordinator = False
    # MG types
    if '̅:̅:' in item:
        sep = '̅:̅:'
        lexical = True
        coordinator = True
    elif ":̅:̅" in item:
        sep = ":̅:̅"
        lexical = True
        coordinator = True
    elif "::" in item:
        sep = "::"
        lexical = True
    elif coord_sep in item:
        sep = coord_sep
        coordinator = True
    elif ":̅" in item:
        sep = ":̅"
        coordinator = True
    else:
        sep = ":"
        coordinator = False

    # split on last instance of separator just in case
    # include spaces
    split = item.rsplit(f" {sep} ", 1)
    features = split[-1].split()
    # log(f"features split into {features}")

    # main = split[0]
    # put main back together in case there's a : in it
    main = f" {sep} ".join(split[0:-1])
    return MGItemProperties(main, lexical, coordinator, features)


class MGItemProperties:
    """
    The parts of the elements that make up an expression.
    Attributes:
        main: str -- l; h; r;
        lexical: bool
        coordinator: bool
        feature_list: list of strings; features of main structure
        interval: Interval where the word comes from in the sentence
    """

    def __init__(self, main, lexical=True, coordinator=False, feature_list=None, interval=None):
        self.main = main
        self.lexical = lexical
        self.coordinator = coordinator
        self.feature_list = feature_list or []
        self.interval = interval

    def __repr__(self):
        sep = ":"
        main = self.main
        if self.lexical:
            sep += ":"
            # only add the index to the leaves
            main += f" {self.interval}"
        if self.coordinator:
            sep += "c"
        return f"{main} {sep} {' '.join(self.feature_list)}"


class MGExpressionProperties:
    """
    The logical parts of a partial result annotation from an annotated tree in MGBank
    Original strings look like this: l; h; r; :: f f f, m : g, m : g h
    main: str -- l; h; r;
    lexical: bool
    coordinator: bool
    feature_list: list of strings; features of main structure
    mover_list: list of MGItemProperties;
    """

    def __init__(self, item_properties: MGItemProperties, mover_list=None):
        self.main = item_properties.main
        self.lexical = item_properties.lexical
        self.coordinator = item_properties.coordinator
        self.feature_list = item_properties.feature_list
        self.mover_list = mover_list or []
        self.interval = item_properties.interval

    def __repr__(self):
        sep = ":"
        main = self.main
        if self.lexical:
            sep += ":"
            main += f" {self.interval}"
        if self.coordinator:
            sep += "c"
        movers = ", ".join(self.mover_list)
        # movers = ""
        # for m in self.mover_list:
        #     movers += f", {m}"
        return f"{main} {sep} {' '.join(self.feature_list)}{movers}"

    @staticmethod
    def annotated_tree_label2mg_expression_properties(label, interval=None):
        """
        Splits the label of an annotated tree from MGBank into its component parts
        @param label: str in the form, below, where :: could also be : or contain
        @param interval: optional interval (i,j) for a lexical item
        the coord_sep, l, h, and r are the left, head, and right of the main item,
         f's are its features, and m's are movers and gs and hs are their features
         l; h; r; :: f f f, m : g, m : g h
        """
        parts = label.split(", ")  # include space
        main = parts[0]
        if len(parts) > 1:
            movers = parts[1:]
        else:
            movers = None

        item_properties = split_item(main)
        item_properties.interval = interval
        if movers:
            movers = [split_item(mover) for mover in movers]

        return MGExpressionProperties(item_properties, movers)


def clean_up_lexical_item_label(full_li_label):
    if re.search(r'ε; .*; ε', full_li_label):  # normal case
        li_label = full_li_label.split(";")[1].strip()
    else:
        li_label = full_li_label  # occasionally trees are printed leaves labelled "word" instead of 'ε; word; ε'
    if li_label == "":  # for LI = ';'
        li_label = ";"
    return li_label


def lexical_constant_maker(label, min_alg: MinimalistAlgebra, conj=False, silent=False):
    """
    Makes a constant AlgebraOp for a minimalist algebra.
    @param min_alg: MinimalistAlgebra being used
    @param label: str: the actual label, e.g. "cat".
    @param inner_algebra: HMAlgebra.
    @param mover_type: Movers class type.
    @param conj: bool: whether this is a conjunction.
    @param silent: bool: whether this should be interpreted as silent.
    @return: AlgebraOp with name = label and function an Expression with inner_item a constant of the given algebra with
                    the given label, or an empty item of the given algebra, and empty movers of the given type
    """
    label = str(label)
    mg_label = label
    if silent:
        inner_item = min_alg.inner_algebra.domain_type()
    else:
        inner_item = min_alg.inner_algebra.domain_type(label)
    if conj:
        mg_label += conj_marker

    return min_alg.constant_maker(word=label, label=mg_label, conj=conj, value=inner_item,
                                  algebra=min_alg.inner_algebra)


def mg_op2mg_op(op: mg.MinimalistFunction, out_alg: hm_algebra.HMAlgebra,
                out_prepare: PreparePackagesHM):
    """
    Turns MinimalistFunction over one algebra into the same one over another algebra, if the inner_op and prepare
     functions of the original function are also in out_alg.
    @param out_prepare: the prepare packages for the output algebra
    @param op: operation to transform.
    @param out_alg: the inner algebra of the new MinimalistFunction.
    @return: MinimalistFunction with inner algebra out_alg, otherwise just like op.
    """

    return mg.MinimalistFunction(op.mg_op,
                                 name=op.name,
                                 inner_op=getattr(out_alg, op.inner_op.__name__) if op.inner_op else None,
                                 from_slot=op.from_slot,
                                 to_slot=op.to_slot,
                                 prepare=getattr(out_prepare, op.prepare.__name__) if op.prepare else None,
                                 adjoin=op.adjoin
                                 )


def get_matching_mover_and_report_silent_mover(movers: list, feat: str):
    """
    Checks to see if a mover is actually silent so we can use Merge1 instead
        of Merge2. MGBank doesn't distinguish covert move from move of something
        accidentally silent, using the simple SMC for silent movers and the DSMC
        for pronounced movers. We want to use only the DSMC,
        so treat these as Merge1
    @param movers: all the movers as a list in the MGBank form
    @param feat: the active feature of the mover we need to check
    @return: bool, list: True iff the mover has no phonetic material, mover features
    """
    found = False
    for mover in movers:
        log(f"checking mover {mover}")
        # look at the first character of each mover's feature string
        f = mover.feature_list[0]
        log(f"searching for {feat} in {f}")
        # we're looking for the matching mover to the active feature
        if f.startswith(f"-{feat}") or f.startswith(f"{feat}~") \
                or feat.lower() == "d" and f.lower() == "d":
            found = True
            break
    if not found:
        return
    if 'ε' in mover.main or re.search(r"^\[.+]$", mover.main):
        # or ("[" in mover_string and "]" in mover_string):  # in {' ε ', f" {epsilon} ", ""}:  #
        log("*** Warning: stealth covert move found: mover is silent")
        return True, mover.feature_list
    else:
        return False, mover.feature_list


# *** main function ***
def nltk_trees2algebra_term(plain: nltk.Tree, annotated: nltk.Tree, synchronous_mg: MinimalistAlgebraSynchronous):
    """
    given two trees, outputs an operation by combining their info to
     determine what algebra operation applies
    Inner algebra: HMStringTripleAlgebra.
    Movers: DSMCMovers.
    Adjoin is different from merge only in that we swap the functor and selectee
        and we add "adjoin" to the function's name. The function will still take
        the head as its first argument and the adjunct as its second.
    Note that because of adjoin we distinguish between functor (which triggers
     the operation) and head. In merge, they are the same, but in adjoin the
     adjunct is the functor even though its selectee remains the head.
    @param plain: nltk.Tree: the one with operation names at internal nodes.
    @param annotated: nltk.Tree: the one with derived items at every node.
    @param synchronous_mg: MinimalistAlgebraSynchronous: the MG to build terms over.
    @return: AlgebraTerm with MinimalistFunctions
    """

    assert_input_is_valid(annotated, plain)

    log(f"\nworking on: {plain.label()} generating {annotated.label()}")

    is_leaf = plain.height() == 1
    if is_leaf:
        return compute_leaf_case_algebra_term(annotated.label(), synchronous_mg)
    else:
        # this actually calls nltk_trees2algebra_term (this function) recursively
        return compute_non_leaf_case_algebra_term_recursively(annotated, plain, synchronous_mg)


# ** MERGES AND MOVES

def compute_non_leaf_case_algebra_term_recursively(annotated: nltk.Tree, plain: nltk.Tree,
                                                   synchronous_mg: MinimalistAlgebraSynchronous):
    """
    Given the two NLTK trees, build the corresponding SynchronousTerm.
    Recursively calls nltk_trees2algebra_term on children.
    @param annotated: nltk.Tree.
    @param plain: nltk.Tree.
    @param synchronous_mg: MinimalistAlgebraSynchronous. The synchronous MG we're building the SynchronousTerm over.
    @return: SynchronousTerm.
    """
    mgbank_operation, plain_children = extract_parent_label_and_children(plain)
    mgbank_partial_result, annotated_children = extract_parent_label_and_children(annotated)

    algebra_op = compute_non_leaf_operation(mgbank_partial_result, mgbank_operation, annotated_children, synchronous_mg)
    if algebra_op:
        log(f"created {algebra_op.name}")  # with properties \n {algebra_op.properties()} \nfrom {annotated.label()}")

    skip_this_operation = algebra_op is None
    if skip_this_operation:
        return skip_identity_function_and_continue_recursion(annotated_children, plain_children, synchronous_mg)

    new_kids = get_zipped_children_in_correct_order(annotated_children, plain_children, mgbank_operation)
    return SynchronousTerm(parent=algebra_op,
                           children=[nltk_trees2algebra_term(t1, t2, synchronous_mg)
                                     # recursive call of top-level function
                                     for t1, t2 in new_kids]
                           )

    # return algebra.AlgebraTerm(algebra_op,
    #                            [nltk_trees2algebra_term(t1, t2)  # recursive call of top-level function
    #                             for t1, t2 in new_kids])


def skip_identity_function_and_continue_recursion(annotated_children, plain_children, synchronous_mg):
    next_annotated, next_plain = skip_identity_function(annotated_children, plain_children)
    return nltk_trees2algebra_term(next_plain, next_annotated, synchronous_mg)


def extract_parent_label_and_children(tree: nltk.Tree):
    return tree.label(), [c for c in tree]


def compute_non_leaf_operation(mgbank_partial_result, mgbank_operation, annotated_children,
                               synchronous_algebra: MinimalistAlgebraSynchronous):
    """

    @param synchronous_algebra: MinimalistAlgebraSynchronous
    @param mgbank_partial_result:
    @param mgbank_operation:
    @param annotated_children:
    @return: MinimalistFunctionSynchronous
    """
    if function_does_nothing(annotated_children, mgbank_operation):
        return

    functor, selectee = get_functor_and_selectee_mgbank_partial_results(annotated_children, mgbank_operation)

    functor_properties = MGExpressionProperties.annotated_tree_label2mg_expression_properties(functor.label())
    # the first feature on the functor determines what happens
    active_feature = functor_properties.feature_list[0]
    log("active: " + active_feature)

    # Move (we've already removed everything that's not overt move)
    if len(annotated_children) == 1:
        operation_properties = compute_move_case(mgbank_operation, active_feature, functor_properties.mover_list)
        if operation_properties is None:
            # we have an identity operation here (silent mover), and just continue
            return

    # merge and adjoin
    elif len(annotated_children) == 2:
        operation_properties = compute_merge_case(active_feature, functor_properties.feature_list, mgbank_operation,
                                                  mgbank_partial_result, functor_properties.mover_list, selectee)
    else:
        raise mg.MGError("Node has more than 2 children")

    if is_adjoin(mgbank_operation):
        operation_properties.adjoin = True

    operation_properties.finalize_properties()
    log(f"operation_properties:\n {operation_properties}")
    algebra_op = operation_properties.get_synchronous_minimalist_function(synchronous_algebra, triple_alg)
    log(f"op chosen: {algebra_op.name}")

    return algebra_op


# ** MERGE

def compute_merge_case(active_feature, functor_features, mgbank_operation, mgbank_partial_result, movers, selectee):
    """
    to figure out if this is merge1 or merge2, we look at the selectee's features.
    If there is a negative licensing feature or right move (cat~) feature after the category, it is merge2
    unless the resulting mover is silent.
    If the category is D, AND parent has a D-mover BUT children don't, then it's also merge2 if the mover isn't silent
    @param active_feature: str
    @param functor_features: str list
    @param mgbank_operation: str from MGBank operation name
    @param mgbank_partial_result: str from MGBank partial result tree
    @param movers: list of partial result strings
    @param selectee: str of MGBank partial result of selected child
    @return: OperationProperties
    """
    # we'll fill this up with properties
    operation_properties = OperationProperties()

    # some special cases
    operation_properties.mg_op, operation_properties.prepare = extract_mg_op_and_prepare_from_mgbank_operation_name(
        mgbank_operation)

    # usual cases
    operation_properties.add_property_if_none("prepare", get_prepare_function(active_feature))
    operation_properties.add_property_if_none("inner_op", get_inner_op(active_feature))

    # i.e. non-functor
    selectee_properties = MGExpressionProperties.annotated_tree_label2mg_expression_properties(selectee.label())
    # i.e. non-head
    other_features, other_movers = set_other_features_and_movers(functor_features, mgbank_operation, movers,
                                                                 selectee_properties)

    is_control = operation_properties.mg_op is None \
                 and active_feature.lower() in {"=d", "d="} and selectee_properties.feature_list[0] == "D"
    if is_control:
        update_control_properties(mgbank_partial_result, movers, other_movers, operation_properties)

    if operation_properties.mg_op is None:
        choose_merge1_or_merge2(mgbank_partial_result, operation_properties, other_features)

    return operation_properties


def extract_mg_op_and_prepare_from_mgbank_operation_name(mgbank_operation: str):
    """
    Some operations can be read straight from the labels
    @param mgbank_operation: str
    @return: (minimalist operation, prepare function)
    """
    min_op, prepare = None, None
    if "_phon" in mgbank_operation:
        min_op = minimalist_algebra_triples.merge1
    # weird head movement operations not in the thesis
    if mgbank_operation == "r_merge_lex":
        min_op = minimalist_algebra_triples.merge1
        prepare = triple_prepare.suffix
    if mgbank_operation == "l_merge_lex" or mgbank_operation == "l_merge_lex_ps":
        min_op = minimalist_algebra_triples.merge1
        prepare = triple_prepare.prefix
    return min_op, prepare


def choose_merge1_or_merge2(mgbank_partial_result, operation_properties, other_features):
    # if we haven't chosen our mg_op yet, look at other's features
    is_merge1 = len(other_features) == 1
    if is_merge1:
        log(f"no more features: merge1")
        operation_properties.mg_op = minimalist_algebra_triples.merge1
    else:
        # merge2, if resulting mover isn't silent
        to_f = get_to_slot_from_feature(operation_properties, other_features)
        # look in the parent node for this mover
        parent_movers = MGExpressionProperties.annotated_tree_label2mg_expression_properties(
            mgbank_partial_result).mover_list
        log(f"parent movers: {parent_movers}")
        matching_mover = get_matching_mover_and_report_silent_mover(parent_movers, to_f)
        if matching_mover is None or matching_mover[0]:
            # we stored something silent, so this is merge1 for us
            log(f"mover is silent: merge1")
            operation_properties.mg_op = minimalist_algebra_triples.merge1
        else:  # otherwise, this is in fact merge2
            operation_properties.mg_op = minimalist_algebra_triples.merge2


def set_other_features_and_movers(functor_features, mgbank_operation, movers, selectee_properties):
    if is_adjoin(mgbank_operation):
        # adjunct is the functor but not the head, and unlike the head's phrase, it might move
        # "other" here == non-head
        other_movers = movers
        other_features = functor_features
    else:
        other_movers = selectee_properties.mover_list
        other_features = selectee_properties.feature_list
    return other_features, other_movers


def get_to_slot_from_feature(operation_properties, other_features):
    to_feat = other_features[1]
    # look up the mover to_slot
    if to_feat.startswith("-"):
        to_f = to_feat[1:].lower()
        operation_properties.to_slot_name = feat2slot[to_f]
    elif to_feat.endswith("~"):
        # print("***Rightward mover***", operation_properties, other_features, to_feat)
        to_f = to_feat[:-1]
        operation_properties.to_slot_name = feat2slot[to_f]
    elif to_feat == "D":
        to_f = to_feat
        operation_properties.to_slot_name = A
    else:
        raise mg.MGError(
            f"can't interpret mover feature {to_feat}"
            f" in {other_features}")
    return to_f


def update_control_properties(mgbank_partial_result, movers, other_movers, operation_properties):
    """
    D can optionally persist, so we need to see if it did
    We update operation_properties only if D persisted, otherwise we do nothing
    @param mgbank_partial_result: str: parent partial result
    @param movers: list of MGItemProperties? for head movers
    @param other_movers: list of MGItemProperties? for non-head movers
    @param operation_properties: OperationProperties found so far for parent
    """
    # check to see if there's now a D-mover in the parent
    parent_movers = MGExpressionProperties.annotated_tree_label2mg_expression_properties(
        mgbank_partial_result).mover_list
    if parent_movers:
        matching_mover = get_matching_mover_and_report_silent_mover(parent_movers, "D")
        d_persisted = matching_mover is not None
        if d_persisted:
            silent, _ = matching_mover
            if not silent:
                if control_mover_was_added_at_this_step(movers, other_movers):
                    operation_properties.mg_op = minimalist_algebra_triples.merge2
                    operation_properties.to_slot_name = A


def control_mover_was_added_at_this_step(head_movers, other_movers):
    """
    Check the children, if any, for existing D movers
    @param head_movers:
    @param other_movers:
    @return: True if children have no D movers
    """
    # max one child has movers
    childrens_movers = head_movers if head_movers else other_movers
    if childrens_movers:
        matching_mover = get_matching_mover_and_report_silent_mover(childrens_movers, "D")
        mover_was_not_previously_there = matching_mover is None
    else:  # no movers so it's definitely new
        mover_was_not_previously_there = True
    return mover_was_not_previously_there


# ** MOVE

def compute_move_case(mgbank_operation, active_feature, movers):
    """
    gets OperationProperties for a Move operation
    @param mgbank_operation: str from MGBank plain tree inner node
    @param active_feature: str
    @param movers: list
    @return: OperationProperties for a Move operation
    """
    if movers is None:
        raise Exception("Can't move if there are no movers")

    move_properties = extract_move_properties_from_active_feature(active_feature)

    move_properties.inner_op = get_move_inner_op(mgbank_operation)

    log("movers: " + str(movers))
    silent, mover_features = get_matching_mover_and_report_silent_mover(movers, move_properties.from_feature)
    if silent:
        # this is marked as regular move, but for us it's covert, so skip it
        return

    choose_move1_or_move2_and_update_to_slot(mgbank_operation, move_properties, mover_features)
    if move_properties.mg_op == minimalist_algebra_triples.move2 and move_properties.to_slot_name is None \
            and move_properties.to_slot is None:
        move_properties.to_slot_name = get_move2_to_slot(mover_features)
    return move_properties


def choose_move1_or_move2_and_update_to_slot(mgbank_operation, operation_properties, mover_features):
    if "_phon" in mgbank_operation:  # covert from now on, so treat it as move1 even if it keeps moving
        operation_properties.mg_op = minimalist_algebra_triples.move1
    log(f"mover_features: {mover_features}")
    if operation_properties.mg_op is None and len(mover_features) > 1:
        operation_properties.mg_op = minimalist_algebra_triples.move2
    elif operation_properties.mg_op is None:
        operation_properties.mg_op = minimalist_algebra_triples.move1


class OperationProperties:
    """
    Properties of minimalist operations to pass around as we figure out the things
     we'll need to build a MinimalistFunction
    from_feature: str
    from_slot: Slot
    to_slot: Slot
    to_slot_name: str
    mg_op: merge1 etc
    inner_op: triple_alg.concat_right etc
    prepare: triple_alg.prepare_simple etc
    adjoin: boolean
    """

    def __init__(self):
        self.from_feature: str or None = None
        self.from_slot: Slot or None = None
        self.successive_cyclic = None
        self.to_slot: Slot or None = None
        self.to_slot_name: str or None = None
        self.mg_op = None
        self.inner_op = None
        self.prepare = None
        self.adjoin = False
        self.reverse = False

    def __repr__(self):
        ret = f"from_feature:{self.from_feature}\n"
        ret += f"from_slot:{self.from_slot}\n"
        ret += f"successive_cyclic:{self.successive_cyclic}\n"
        ret += f"to_slot:{self.to_slot}\n"
        ret += f"to_slot_name:{self.to_slot_name}\n"
        ret += f"mg_op:{self.mg_op}\n"
        ret += f"inner_op:{self.inner_op}\n"
        ret += f"prepare:{self.prepare}\n"
        ret += f"adjoin:{self.adjoin}\n"
        return ret

    def add_property_if_none(self, property_name, value):
        if getattr(self, property_name) in {None, False}:
            setattr(self, property_name, value)

    def get_minimalist_function(self):
        """
        Uses the functions, slots, and adjoin to get a MinimalistFunction
        @return: MinimalistFunction
        """
        return mg.MinimalistFunction(self.mg_op,
                                                inner_op=self.inner_op,
                                                from_slot=self.from_slot,
                                                to_slot=self.to_slot,
                                                prepare=self.prepare,
                                                adjoin=self.adjoin,
                                     reverse=self.reverse
                                                )

    def get_synchronous_minimalist_function(self, synchronous_algebra: MinimalistAlgebraSynchronous, inner_algebra: Algebra):
        """
        Uses the functions, slots, and adjoin to get a MinimalistFunctionSynchronous
        @return: MinimalistFunctionSynchronous
        """
        inner_op = self.inner_op.__name__ if self.inner_op is not None else None
        prepare = self.prepare.__name__ if self.prepare is not None else None
        inner_ops = {inner_algebra: InnerAlgebraInstructions(inner_op, prepare, reverse=self.reverse)}

        return MinimalistFunctionSynchronous(synchronous_algebra,
                                                self.mg_op,
                                                inner_ops=inner_ops,
                                                from_slot=self.from_slot,
                                                to_slot=self.to_slot,
                                                adjoin=self.adjoin,
                                                )

    def finalize_properties(self):
        """
        Adds default properties in default cases
        """
        self.finalize_to_slot()
        self.finalize_prepare()
        self.finalize_inner_op()

    def finalize_to_slot(self):
        """
        Ensures merge1 and move1 have no to-slot and converts to_feature string into to_slot Slot
        """
        if self.mg_op in {minimalist_algebra_triples.merge1, minimalist_algebra_triples.move1}:
            self.to_slot = None
        elif self.to_slot is None:
            self.to_slot = slot_name2slot(self.to_slot_name)

    def finalize_from_slot(self):
        if self.mg_op in {minimalist_algebra_triples.merge1, minimalist_algebra_triples.merge2}:
            self.from_slot = None

    def finalize_prepare(self):
        """
        Adds default prepare_simple to Merge if none is given
        """
        # if self.prepare is None and self.mg_op in {minimalist_algebra_triples.merge1,
        #                                            minimalist_algebra_triples.merge2}:
        #     self.prepare = triple_prepare.simple
        pass

    def finalize_inner_op(self):
        """
        Ensures Move2 has no inner op (none ever used)
        Ensures Merge2 has concat_right (always just concat right since even with HM leftovers are empty)
        adds default concat_left for Move1 if no inner_op given
        """
        if self.mg_op in {minimalist_algebra_triples.move2, minimalist_algebra_triples.merge2}:
            self.inner_op = None
        elif self.mg_op == minimalist_algebra_triples.move1 and self.inner_op is None:
            self.inner_op = triple_alg.concat_left
        if self.mg_op == minimalist_algebra_triples.merge2 and self.inner_op is None:
            self.inner_op = triple_alg.concat_right
        elif self.mg_op == minimalist_algebra_triples.merge1 and self.inner_op is None:
            raise Exception(
                f"*** Merge1 with no inner op!")


def get_move2_to_slot(mover_features: list[str]):
    """
    extract the bare feature e.g. wh from the second feature in the list
    (the first feature is the from-slot)
    this will be our to_feat for move2
    @param mover_features: string list
    @return: Slot
    """
    to_full_feat = mover_features[1]  # e.g. -wh or v~ or D
    if to_full_feat.startswith("-"):
        to_feat = to_full_feat[1:].lower()
    elif to_full_feat.endswith("~"):
        to_feat = to_full_feat[:-1].lower()
    elif to_full_feat == "D":
        to_feat = "D"
    else:
        raise mg.MGError(f"Can't interpret feature {to_full_feat}")
    log(f"to_feat: {to_feat}")
    to_slot = feat2slot[to_feat]  # look up the to_slot
    return to_slot


def polarity_indicates_normal_move(feature):
    return feature.startswith("+")


def polarity_indicates_control_move(feature):
    """
    Control D being selected; treated as move
    @param feature: str
    @return: boolean
    """
    return feature.lower() in {"=d", "d="}


def extract_move_properties_from_active_feature(active_feature):
    if polarity_indicates_normal_move(active_feature):
        return get_normal_move_properties(active_feature)
    elif polarity_indicates_control_move(active_feature):
        return get_control_move_properties(active_feature)
    else:
        return get_rightward_category_move_properties(active_feature)


def get_rightward_category_move_properties(from_feat):
    properties = OperationProperties()
    properties.from_slot = R
    properties.from_feature = from_feat.lower()
    return properties


def get_control_move_properties(from_feat: str):
    feat = from_feat.replace("=", "")
    properties = OperationProperties()
    properties.from_feature = feat.lower()
    properties.from_slot = A
    return properties


def feature_ends_with_unneeded_marker(feature):
    return feature.endswith("!")


def feature_marked_as_successive_cyclic(feature):
    return feature.endswith("?")


def remove_feature_suffix(feature):
    """feature suffixes are always of length 1"""
    return feature[:-1]


def remove_feature_polarity(feature: str, polarity_to_left=True):
    """
    Remove eg + from +f, or = from f=, returning f
    @param feature: string representation of feature
    @param polarity_to_left: re position of the polarity marker
    @return: str
    """
    if polarity_to_left:
        return feature[1:]
    else:
        return feature[:-1]


def get_normal_move_properties(from_feat: str):
    """
    For non-control, leftward movement, including successive-cyclic
    @param from_feat: str
    @return: OperationProperties
    """
    properties = OperationProperties()
    from_feat = remove_feature_polarity(from_feat).lower()
    if feature_ends_with_unneeded_marker(from_feat):
        from_feat = remove_feature_suffix(from_feat)
    if feature_marked_as_successive_cyclic(from_feat):
        from_feat = remove_feature_suffix(from_feat)
        properties.successive_cyclic = True
        log("successive cyclic movement")
        # we'll put the mover back in its original to_slot since it's successive-cyclic move
        properties.to_slot_name = feat2slot[from_feat]
        properties.mg_op = minimalist_algebra_triples.move2
    properties.from_feature = from_feat
    properties.from_slot = feat2slot[properties.from_feature]
    return properties


def get_move_inner_op(mgbank_operation):
    if mgbank_operation.startswith("r_"):
        return triple_alg.concat_right
    elif mgbank_operation.startswith("l_"):
        return triple_alg.concat_left
    else:
        raise Exception(f"can't interpret move function {mgbank_operation}")


def get_prepare_function(active_feature):
    """
    Uses polarity2prepare dict to find corresponding prepare function
    @param active_feature: str
    @return: a prepare function from triple_alg, or None if none is found
    """
    prepare = None
    for pattern in polarity2prepare:
        if pattern.search(active_feature):
            prepare = polarity2prepare[pattern]
            if prepare is not None:
                log("prepare op found: " + prepare.__name__)
            break
    return prepare


def get_inner_op(active_feature):
    """
    Uses polarity2inner_op dict to find corresponding inner op
    @param active_feature: str
    @return: an operation from triple_alg, or None if none is found
    """
    inner_op = None
    for pattern in polarity2inner_op:
        if pattern.search(active_feature):
            inner_op = polarity2inner_op[pattern]
            log("inner_op found: " + inner_op.__name__)
            break
    return inner_op


def skip_identity_function(annotated_children, plain_children):
    # these are always unary, so index 0 gets the one child
    next_plain, next_annotated = plain_children[0], annotated_children[0]
    return next_annotated, next_plain


def function_does_nothing(annotated_children, mgbank_operation):
    # we skip functions that don't do anything, and we also skip covert move
    return is_ident_function(len(annotated_children), mgbank_operation) or mgbank_operation == "c_move"


def get_functor_and_selectee_mgbank_partial_results(annotated_children, mgbank_operation):
    """
    Given the partial results for the children and the operation from MGBank, figure out which is the functor.
    Return the children in functor, other order.
    @param annotated_children: strings, I think, but it doesn't matter.
    @param mgbank_operation: str (operation name).
    @return: functor, other (or functor, None if unary).
    """
    # check if the functor is on the right
    if functor_right(mgbank_operation):
        functor = annotated_children[1]  # the thing that decides on the operation
        selectee = annotated_children[0]  # the other thing
    else:
        functor = annotated_children[0]
        if len(annotated_children) == 2:
            selectee = annotated_children[1]
        else:
            selectee = None
    return functor, selectee


def get_zipped_children_in_correct_order(annotated_children, plain_children, mgbank_operation):
    swapped = head_right(mgbank_operation)
    new_kids = list(zip(plain_children, annotated_children))
    if swapped:
        new_kids.reverse()
    return new_kids


# leaf case functions

def compute_leaf_case_algebra_term(annotated_tree_label, synchronous_mg: MinimalistAlgebraSynchronous):
    log("LI")  # LI means lexical item
    leaf_properties = MGExpressionProperties.annotated_tree_label2mg_expression_properties(annotated_tree_label)
    logger.debug(f"leaf_properties: {leaf_properties}")

    # Newer code (frederieke):
    # if label_is_silent(leaf_properties.main):
    #    return
    #    return skip_identity_function_and_continue_recursion(annotated_children, plain_children)
    # End of New Code

    log(leaf_properties.main)
    li_label = clean_up_lexical_item_label(leaf_properties.main)
    # log(f"label {li_label}")
    if leaf_properties.coordinator:
        log("+coord")
    li = lexical_constant_maker(li_label, synchronous_mg, conj=leaf_properties.coordinator,
                                silent=label_is_silent(li_label))
    return SynchronousTerm(li)
    # return synchronous_mg.synchronous_term_from_minimalist_term(algebra.AlgebraTerm(algebra_li), triple_alg)
    # return algebra.AlgebraTerm(algebra_li)


def label_is_silent(li_label: str):
    # MGbank silent items have labels like "[past]"
    return li_label.startswith("[")


def assert_input_is_valid(annotated, plain):
    assert plain.height() == annotated.height()




def clean_original_sentence(node_label):
    """
    Some sentences in MGBank are in triple format, instead of the sentence proper.
    """
    original_sentence = \
        MGExpressionProperties.annotated_tree_label2mg_expression_properties(node_label).main

    if re.search(r'ε; .*; ε', original_sentence):
        original_sentence_list = [original_sentence.split("; ")[1]]
    else:
        original_sentence_list = original_sentence.split()
        original_sentence_list = [w for w in original_sentence_list if not w.startswith("[")]
    return original_sentence_list


def add_addresses(original_term: SynchronousTerm, hm_algebra: HMAlgebra, address_algebra: HMTripleAlgebra):
    """
    Adds interpretation with Triples over address lists.
    @param original_term: SynchronousTerm to update.
    @param hm_algebra: the HMTripleAlgebra we already have in the tree.
    @param address_algebra: the HMTripleAlgebra over addresses to add.
    @return: updated term
    """

    def _add_addresses(term: SynchronousTerm, address):

        logger.debug("term: " + str(term))

        if term.is_leaf():
            # for these purposes, we don't want the silent heads to have addresses
            if len(term.spellout(hm_algebra)) == 0:
                new_object = Triple(component_type=list)
            else:
                new_object = Triple([[address]])

            # If there are no specified inner ops for any existing interpretations
            # (because they'll just use the default), initialise an empty dict
            if term.parent.inner_ops is None:
                term.parent.inner_ops = {}

            # assert isinstance(term.parent.function, Expression), f"{term.parent.function} is not an Expression"
            if hm_algebra in term.parent.inner_ops:
                new_name = term.parent.inner_ops[hm_algebra].op_name
            else:
                new_name = term.parent.name
            if address_algebra in term.parent.inner_ops:
                # we need to copy this leaf
                new_inner_ops = {}
                for a in term.parent.inner_ops:
                    if a != address_algebra:
                        new_inner_ops[a] = deepcopy(term.parent.inner_ops[a])
                # algebra functions need unique names so we add the address to the copy
                new_name = term.parent.name  # + "_" + address
                new_function = MinimalistFunctionSynchronous(minimalist_algebra=term.parent.minimalist_algebra,
                                                             minimalist_function=term.parent.minimalist_function,
                                                             inner_ops=new_inner_ops,
                                                             conj=term.parent.conj,
                                                             name=new_name)
                new_function.inner_ops[address_algebra] = InnerAlgebraInstructions(new_name, leaf_object=new_object)
                return SynchronousTerm(new_function), address

            term.parent.inner_ops[address_algebra] = InnerAlgebraInstructions(new_name, leaf_object=new_object)
            return term, address
        else:
            if term.parent.inner_ops is not None:
                # logger.debug(f"\n############# Inner Ops {term.parent.inner_ops} #####################\n")
                assert hm_algebra in term.parent.inner_ops
                # these are both HM algebras so their function names etc are the same
                term.parent.inner_ops[address_algebra] = term.parent.inner_ops[hm_algebra]
            new_pairs = [_add_addresses(kid, address + str(i)) for (i, kid) in
                         enumerate(term.children)]
            # we just want the new_term, not the subterm of the other term
            term.children = [pair[0] for pair in new_pairs]
            # return the current updated subterm and other subterm
            return term, address

    return _add_addresses(original_term, "")[0]


def find_index(address, sentence):
    """
    Look through a list of lists of addresses and fine the index in the main list that contains the given address.
    Used to find where in the sentence a given leaf in the MG term actually ends up.
    Copying from ATB movement yields some words with multiple sources,
     which is why the "sentence" is a list of lists of addresses instead of just a list of addresses.
    @param address: any, but used for str (tree address as string of integers).
    @param sentence: list of lists of addresses.
    @return: int, the index where the address was found.
    """
    for i, addresses in enumerate(sentence):
        if address in addresses:
            return i


def add_intervals(original_term: SynchronousTerm, sentence, interval_algebra: HMIntervalPairsAlgebra, address_algebra=address_alg):
    """
    Given a SynchronousTerm that includes an address triple algebra,
    add an Interval algebra with the indices in the string.
    @param original_term: SynchronousTerm that includes an address triple algebra.
    @param sentence: spellout of the term using the address triple algebra, so list of lists of strings.
    @param interval_algebra: HMIntervalPairsAlgebra we're adding.
    @return: updated term
    """

    def _add_intervals(term: SynchronousTerm):
        """
        Recursive inner function.
        @param term: current subterm.
        @return: updated subterm.
        """
        if term.is_leaf():
            address_list = term.parent.inner_ops[address_algebra].leaf_object.head
            if len(address_list) > 0:
                # these are always unary list in the tree, because it's the address of the leaf itself.
                start = find_index(address_list[0][0], sentence)
                name = str(start)

            else:
                start = None
                name = term.parent.name
            term.parent.inner_ops[interval_algebra] = InnerAlgebraInstructions(name, leaf_object=PairItem(start))
            return term
        else:
            if term.parent.inner_ops is not None and address_algebra in term.parent.inner_ops:
                # these are both HM algebras, so their ops and prepares have the same names
                term.parent.inner_ops[interval_algebra] = term.parent.inner_ops[address_algebra]
            term.children = [_add_intervals(kid) for kid in term.children]
            # return the current updated subterm and other subterm
            return term

    return _add_intervals(original_term)


# def addressed_term2action_list(term: algebra.AlgebraTerm, sentence):
#     """
#     TODO update
#     Get a list of parser steps from a term with addresses and a sentence with addresses
#     Currently, we put the new item in the head's place in the stack.
#     If the head is silent, we call it epsilon and we put the new item in the other's place in the stack
#     If both are silent, we tack the new item onto the end of the stack
#     If there is ATB movement, there is more than one address for a given word. We append "+keep" to the function name.
#     @param term: AlgebraTerm with MinimalistFunctions over HMPairIntervalAlgebra and ListMovers
#     @param sentence: list of (str, [addresses]) pairs, where the addresses are strings of 0's and 1's
#     @return: list of actions in the form (MinimalistFunction, list of children's stack indices)
#     """
#
#     address_actions = []  # actions with tree addresses of arguments rather than strings or stack indices
#
#     def subtree2action(t: algebra.AlgebraTerm, parent_address):
#         """
#         Go through the term top down and add actions to the list of actions in the form (operation, [child addresses]
#         @param t: AlgebraTerm over MinimalistFunctions with HMStringTripleAlgebra and ListMovers
#         @param parent_address: list of strings of 0's and 1's
#         @return: nothing, just update address_actions
#         """
#         if t.children is not None:
#             # move or merge
#             child_addresses = [parent_address + '0']
#             if len(t.children) == 2:  # merge
#                 child_addresses.append(parent_address + '1')
#             # update list initialised above
#             address_actions.append((t.parent, child_addresses))
#             for i, child in enumerate(t.children):  # recurse
#                 subtree2action(child, parent_address + str(i))
#
#     # fill up the address action list
#     subtree2action(term, "")  # root address is the empty string
#     address_actions.reverse()  # we want these backwards because we work bottom-up
#     actions = []  # the actions with stack indices
#
#     log("\n** sentence and addresses\n")
#     log(str(sentence))
#     stack = [addresses for _, addresses in sentence]  # initialise stack with all addresses of each word
#
#     # turn the actions with addresses into actions with stack indices
#     for (op, kids) in address_actions:
#         log(f"from {op}, {kids}")
#         children = []
#         for i, addresses in enumerate(stack):
#             if kids[0] in addresses:
#                 children.append(i)
#                 addresses.remove(kids[0])
#                 addresses.append(kids[0][:-1])  # replace the head with its parent on the stack
#                 break
#         if len(children) == 0:  # if it wasn't in the sentence, it's silent
#             children.append(epsilon_label)
#         if len(kids) == 2:
#             # merge
#             for i, addresses in enumerate(stack):
#                 if kids[1] in addresses:  # non-head
#                     children.append(i)
#                     addresses.remove(kids[1])
#                     if len(addresses) > 0:
#                         op.name += "+keep"  # if there are more copies, we signal that we need to keep it on the stack
#                     if epsilon_label in children:  # if head is silent, keep this stack location
#                         addresses.append(kids[0][:-1])
#                     if len(addresses) == 0:  # remove from stack if empty
#                         stack.remove(addresses)
#                     break
#
#             if len(children) == 1:
#                 children.append(epsilon_label)
#                 if children == [epsilon_label, epsilon_label]:  # if both silent, append to stack
#                     stack.append([kids[0][:-1]])
#
#         actions.append((op, children))
#         log(f"Stack: {stack}")
#         log(f"New Action: {(op, children)}")
#
#     # make them into HMIntervalPair ops
#     return [(mg_op2mg_op(operation, pair_alg, pair_prepare), kids) for (operation, kids) in actions]


def nltk_trees2term(plain_tree, annotated_tree):
    """
    Given a pair of nltk trees read in from file, generates a SynchronousTerm with inner algebras
    string triples, address triples, and interval pairs.
    @param plain_tree: nltk.Tree: MGBank derivation tree with operation names and features.
    @param annotated_tree: nltk.Tree: MGBank derivation tree with partial results.
    @return: SynchronousTerm with strings, addresses, and intervals.
    """
    original_sentence = clean_original_sentence(annotated_tree.label())
    outcome = nltk_trees2algebra_term(plain_tree, annotated_tree, smg)

    # add addresses
    smg.inner_algebras[address_alg] = address_prepare
    add_addresses(outcome, triple_alg, address_alg)

    # use addresses to add Intervals
    address_output = outcome.spellout(address_alg)
    triple_output = outcome.spellout(triple_alg).split()

    if len(triple_output) != len(address_output):
        raise MGError("ERROR: Addresses output does not match string output")
    if original_sentence != triple_output:
        logger.error("ERROR: original sentence does not match derived sentence")
        logger.error(f"orig: {original_sentence}")
        logger.error(f"deri: {triple_output}")
        raise MGError(f"derived sentence does not match original")
    smg.inner_algebras[pair_alg] = pair_prepare
    add_intervals(outcome, address_output, pair_alg)

    # test Intervals
    outcome.spellout(pair_alg)
    return outcome


if __name__ == "__main__":

    from ..minimalism.mgbank_input_codec import read_corpus_file, CONJ
    import sys

    # from term_output_codec import term2nltk_tree

    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    else:
        corpus_path = "./data/processed/mg_bank/split/"
        # corpus_path = "./data/raw_data/MGBank/"


    smg = MinimalistAlgebraSynchronous(inner_algebras=[triple_alg], prepare_packages=[triple_prepare],
                                       mover_type=DSMCMovers)
    print("** SMG **")
    print(smg)

    # directory2file(path, "corpora/seeds/")

    file = "09/wsj_0927.mrg"

    # just do one sentence
    file_number = "0006"
    # sentence number within file (just the order, not the actual number in the file I think)
    i = 2
    sub_corpus = "dev"  # "wsj_MGbankSeed"    # "dev"
    corpus_path += sub_corpus

    smc = 0
    atb = 0
    other_errors = []
    ok = 0
    terms = []

    for dir in os.listdir(corpus_path):
        dir_path = os.path.join(corpus_path, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(".mrg"):
                    in_path = os.path.join(corpus_path, dir, file)
                    tree_pairs = read_corpus_file(in_path)
                    for k, (p, a) in enumerate(tree_pairs):
                        try:
                            t = nltk_trees2term(p, a)
                            terms.append(t)
                            ok += 1
                        except SMCViolation as e:
                            logger.warning(f"in {file}, entry {k}: {e}")  # , stack_info=True)
                            smc += 1
                        except ATBError as e:
                            logger.warning(f"in {file}, entry {k}: {e}")  # , stack_info=True)
                            atb += 1
                        except Exception as e:
                            logger.error(f"in {file}, entry {k}: {e}", stack_info=True)
                            other_errors.append(e)
                            # raise e
    print("smc and atb errors", smc, atb)
    print(f"other errors: {other_errors}")
    print(f"ok:     {ok}")

    dill.dump(terms, open(f"./data/processed/{sub_corpus}.pickle", 'wb'))

    # in_path = f"{corpus_path}/{file_number[:2]}/wsj_{file_number}.mrg"
    #
    # tree_pairs = read_corpus_file(in_path) #[i:i+1]
    # tree_pairs = read_corpus_file(f"{corpus_path}/{file}")
    # tree_pairs[0][1].draw()


    #   other_triple_alg = hm_triple_algebra.HMTripleAlgebra(name="other triple alg")

    # def replace_inner_algebra(t: AlgebraTerm):
    #     """
    #     Just for a quick way to have two terms to test with
    #     """
    #     if t.is_leaf():
    #         return t
    #     else:
    #         t.parent = mg_op2mg_op(t.parent, other_triple_alg, triple_prepare)
    #
    #         t.children = [replace_inner_algebra(child) for child in t.children]
    #         return t

    # print()
    # for k, (p, a) in enumerate(tree_pairs):
    #     original_sentence = clean_original_sentence(a.label())
    #     outcome = nltk_trees2algebra_term(p, a, smg)
    #     # outcome, s = nltk_trees2addressed_list_term(p, a)
    #     print(f"\n************** Sentence {k} **********************")
    #     print(a.label())
    #     # print(outcome)
    #     inner_outcome = outcome.interp(triple_alg)
    #     # print("inner term:", inner_outcome)
    #     # triple = inner_outcome.inner_term.evaluate()
    #     # print(triple)
    #     # print(outcome.spellout(triple_alg))
    #
    #     smg.inner_algebras[address_alg] = address_prepare
    #     add_addresses(outcome, triple_alg, address_alg)
    #
    #     address_output = outcome.spellout(address_alg)
    #     triple_output = outcome.spellout(triple_alg).split()
    #     if len(triple_output) != len(address_output):
    #         print("ERROR: Addresses output does not match string output")
    #     # print(triple_output)
    #     # print(address_output)
    #     # print(original_sentence)
    #     if original_sentence != triple_output:
    #         print("ERROR: original sentence does not match derived sentence")
    #         print(f"orig: {original_sentence}")
    #         print(f"deri: {triple_output}")
    #
    #     smg.inner_algebras[pair_alg] = pair_prepare
    #     add_intervals(outcome, address_output, pair_alg)
    #
    #     # interval_term = outcome.interp(pair_alg).inner_term
    #     # inner_outcome.inner_term.to_nltk_tree().draw()
    #     # interval_term.to_nltk_tree().draw()
    #
    #     intervals = outcome.spellout(pair_alg)
    #     # print(intervals)





        # other = outcome.to_minimalist_term(triple_alg)
        # print("other", other)
        # print(other.evaluate())

        #
        #
        # new_tree = replace_inner_algebra(other)
        # print(new_tree.evaluate().spellout())
        #
        # x = outcome.add_algebra(other_triple_alg, new_tree)
        # print("new one", x)
        # print(x.parent.inner_ops)
        # #
        # print(x.interp(other_triple_alg).spellout() == x.interp(triple_alg).spellout())
        #

        # print(s)

#         acts = addressed_term2action_list(outcome, s)
#     print("\nActions:")
#     for j, act in enumerate(acts):
#         print(f"{j}. {act}")
#
# #
# keeps = []
# errors = []
#
# for dir in os.listdir(corpus_path):
#     if os.path.isdir(f"{corpus_path}/{dir}"):
#         for f in os.listdir(f"{corpus_path}/{dir}"):
#             if f.endswith(".mrg"):
#                 log(f"\nFile: {f}")
#                 tree_pairs = read_corpus_file(f"{corpus_path}/{dir}/{f}")
#
#                 for i, (p, a) in enumerate(tree_pairs):
#                     try:
#                         outcome, s = nltk_trees2addressed_list_term(p, a, f, i)
#                         acts = addressed_term2action_list(outcome, s)
#                         for act in acts:
#                             if '+keep' in act[0].name:
#                                 keeps.append((f, i, s, acts))
#                                 break
#                     except Exception as e:
#                         errors.append((f, i, e))
#
# for item in keeps:
#     print(item)
#
# print(f"Number of items with keep: {len(keeps)}")
#
# print(f'number of errors: {len(errors)}')
