import logging

from data_reader import MOVE, MERGE, is_merge, is_move, is_shift
from minimalist_parser.algebras.algebra import AlgebraTerm
from minimalist_parser.algebras.algebra_objects.intervals import IntervalError, PairItem
from minimalist_parser.algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from minimalist_parser.algebras.string_algebra import BareTreeStringAlgebra
from minimalist_parser.convert_mgbank.slots import A, Abar, R, Self
from minimalist_parser.convert_mgbank.term2actions import SHIFT, ARG_CODE_OPARGSHIFT, SHIFT_ARG_CODE_OPARGSHIFT, OP_CODE_OPARGSHIFT
from minimalist_parser.minimalism.mg_errors import SMCViolation, MGError
from minimalist_parser.minimalism.minimalist_algebra import Expression
from minimalist_parser.minimalism.minimalist_algebra_synchronous import MinimalistAlgebraSynchronous, SynchronousTerm, \
    MinimalistFunctionSynchronous, InnerAlgebraInstructions
from minimalist_parser.minimalism.movers import DSMCMovers, Movers
from minimalist_parser.minimalism.prepare_packages.interval_prepare_package import IntervalPairPrepare
from minimalist_parser.minimalism.prepare_packages.prepare_packages_bare_trees import PreparePackagesBareTrees



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

# distinguish whether we can choose a silent head
SILENT = "silent"
STACK_ITEM = "stack_item"

# string algebra over bare trees as algebra terms
string_alg = BareTreeStringAlgebra(name="string_algebra")
string_alg.add_constant_maker()

# interval algebra for parse items
interval_algebra = HMIntervalPairsAlgebra()
interval_prepare = IntervalPairPrepare()
interval_algebra.add_constant_maker()

# Synchronous algebra over both
mg = MinimalistAlgebraSynchronous([string_alg, interval_algebra],
                                  prepare_packages=[PreparePackagesBareTrees(name="for strings",
                                                                             inner_algebra=string_alg),
                                                    interval_prepare], mover_type=DSMCMovers)



def legal_next_sr_item(sr_sequence: list):
    """
    Given a shift-reduce sequence, so far, what are we allowed to do?
    @param sr_sequence: list of strings, operation names or stack indices or buffer indices
    @return: 0, 1, or 2, according to whether we need an operation, a stack/silent argument,
                or something from the buffer (for SHIFT)
    """
    if len(sr_sequence) == 0:
        return OP_CODE_OPARGSHIFT
    elif is_shift(sr_sequence[-1]):
        return SHIFT_ARG_CODE_OPARGSHIFT
    elif len(sr_sequence) >= 2 and is_move(sr_sequence[-2]) or is_shift(sr_sequence[-2]):
        # unary shift and move mean we need an operation after 1 arg
        return OP_CODE_OPARGSHIFT
    elif len(sr_sequence) >= 3 and is_merge(sr_sequence[-3]):
        # binary merge means we need an operation after 2 args
        return OP_CODE_OPARGSHIFT
    else:
        return ARG_CODE_OPARGSHIFT


def legal_next_sr_item_constrained(sr_sequence: list):
    """
    Given a shift-reduce sequence, so far, what are we allowed to do?
    @param sr_sequence: list of strings, operation names or stack indices or buffer indices
    @return: list; first element is 0, 1, or 2, according to whether we need an operation, a stack/silent argument,
                or something from the buffer (for SHIFT).
                following elements are strings saying which operations or arguments are actually possible,
                 given the configuration of the SR sequence.
                 (Note we're not checking here what operations are legal in the partial minimalist algebra)
    """
    if len(sr_sequence) == 0:
        return [OP_CODE_OPARGSHIFT, MERGE, SHIFT]
    elif is_shift(sr_sequence[-1]):
        return [SHIFT_ARG_CODE_OPARGSHIFT]
    elif len(sr_sequence) >= 2 and is_move(sr_sequence[-2]) or is_shift(sr_sequence[-2]):
        # unary shift and move mean we need an operation after 1 arg
        return [OP_CODE_OPARGSHIFT, MERGE, MOVE, SHIFT]
    elif len(sr_sequence) >= 3 and is_merge(sr_sequence[-3]):
        # binary merge means we need an operation after 2 args
        return [OP_CODE_OPARGSHIFT, MERGE, MOVE, SHIFT]
    elif len(sr_sequence) >= 2 and is_move(sr_sequence[-1]):
        # Move can't be followed by a silent head
        return [ARG_CODE_OPARGSHIFT, STACK_ITEM]
    else:
        # first or second Merge argument
        return [ARG_CODE_OPARGSHIFT, STACK_ITEM, SILENT]


def apply_minimalist_operation(operation_name: str, children: list[SynchronousTerm]):
    """
    Use the operation on the children. If
    @param operation_name: str: merge or move neme to look up in mg.ops
    @param children: list of SynchronousTerms
    @return: ParseItem of the new term and its interpretation into the interval algebra, with the string interp as backup
    """
    new_term = SynchronousTerm(mg.ops[operation_name], children)
    pair_item = string = None
    try:
        expression = new_term.interp(interval_algebra)
        try:
            pair_item = expression.inner_term.evaluate()
        except IntervalError:
            pair_item = None
        try:
            string = new_term.interp(string_alg)
        # TODO we might need different errors here, but for now let's print so we can make sure this works when it should
        except Exception as e:
            logger.warning(f"Exception raised while trying to apply {operation_name}: {e}")
            string = None
    except MGError as e:
        logger.warning(f"MGError {e} raised applying {operation_name}")
        expression = None

    return ParseItem(new_term, expression, pair_item, string)




class ParseItem:
    def __init__(self, term: SynchronousTerm, expression: Expression, pair_item: PairItem, string: str):
        """
        Partial results
        @param pair_item: expression.inner_term.evaluate()
        @param expression: term.interp(interval_algebra)
        @param term: SynchronousTerm
        @param string: term.interp(string_alg).inner_term.evaluate()
        """
        self.pair_item = pair_item
        self.expression = expression
        self.term = term
        self.string = string

    def is_valid(self):
        return self.pair_item is not None

    def is_valid_for_strings(self):
        return self.string is not None

    def valid_mover_handling(self):
        return self.expression is not None

    def is_complete(self):
        if self.is_valid():
            return self.expression.is_complete()
        # otherwise just worry about movers
        if self.valid_mover_handling():
            return not self.expression.has_movers()
        else:
            return False

    def possible_operations(self):
        """
        Finds all possible operations that can be performed, without knowing what the other argument is, in the case of Merge.
        Move1 includes the outcome as a ParseItem.

        @return: dict from
        merge1 to MinimalistFunctionSynchronous list,
        merge2 to MinimalistFunctionSynchronous list,
        move1 to MinimalistFunctionSynchronous, ParseItem pair list,
        move2 to MinimalistFunctionSynchronous list
        """
        ret = {"merge1": [], "merge2": [], "move1": [], "move2": []}

        ret["merge1"].append(MinimalistFunctionSynchronous(mg, mg.merge1, {string_alg: InnerAlgebraInstructions("concat_left", reverse=True),
                                                                           interval_algebra: InnerAlgebraInstructions("concat_left"),}))
        if self.pair_item.can_hm:
            for prepare in ["suffix", "prepare", "hm_atb", "excorporation"]:
                ret["merge1"].append(
                    MinimalistFunctionSynchronous(mg, mg.merge1, {string_alg: InnerAlgebraInstructions(prepare=prepare),
                                                                  interval_algebra: InnerAlgebraInstructions(
                                                                      prepare=prepare)}))

        from_slots = self.expression.movers.mover_dict.keys()
        to_slots = [slot for slot in mg.slots if slot not in from_slots]
        for to_slot in to_slots:
            for from_slot in from_slots:
                ret["move2"].append(MinimalistFunctionSynchronous(mg, mg.move2, from_slot=from_slot, to_slot=to_slot))
            ret["merge2"].append(MinimalistFunctionSynchronous(mg, mg.merge2, to_slot=to_slot))
            ret["merge2"].append(MinimalistFunctionSynchronous(mg, mg.merge2, to_slot=to_slot, adjoin=True))

            if self.pair_item.can_hm:
                for prepare in ["suffix", "prepare", "hm_atb", "excorporation"]:
                    ret["merge2"].append(MinimalistFunctionSynchronous(mg, mg.merge2, {string_alg: InnerAlgebraInstructions(prepare=prepare),
                                                                                       interval_algebra: InnerAlgebraInstructions(prepare=prepare)}, to_slot=to_slot))


        for from_slot in from_slots:
            try:
                op = MinimalistFunctionSynchronous(mg, mg.move1, from_slot=from_slot,
                                                   inner_ops={string_alg: InnerAlgebraInstructions("concat_left", reverse=True),
                                                              interval_algebra: InnerAlgebraInstructions("concat_left")
                                                              })
                t = SynchronousTerm(op, [self.term])
                e = t.interp(interval_algebra)
                i = e.inner_term.evaluate()
                s = t.interp(string_alg).inner_term.evaluate()
                ret["move1"].append((op, ParseItem(t, e, i, s)))
            except:
                pass

            try:
                op = MinimalistFunctionSynchronous(mg, mg.move1, from_slot=from_slot,
                                                   inner_ops={string_alg: InnerAlgebraInstructions("concat_right"),
                                                              interval_algebra: InnerAlgebraInstructions("concat_right")
                                                              })
                t = SynchronousTerm(op, [self.term])
                e = t.interp(interval_algebra)
                i = e.inner_term.evaluate()
                s = t.interp(string_alg).inner_term.evaluate()
                ret["move1"].append((op, ParseItem(t, e, i, s)))
            except:
                pass














# class Feature:
#     """
#     Making a feature class mostly so we have a type to refer to
#     """
#     def __init__(self, f):
#         self.feature = f
#
#
#     def __str__(self):
#         return str(self.feature)
#
#     __repr__ = __str__
#
#     def __eq__(self, other):
#         """
#         For some reason, same features aren't showing up as equal to each other without this
#         @param other: another feature
#         @return: true iff they're both features and have the same feature value
#         """
#         if type(other) == Feature:
#             return self.feature == other.feature
#         else:
#             return False
#
#     def __hash__(self):
#         """
#         no idea, need to make __eq__ work. Could make a "hash" of things, haha!
#         @return:
#         """
#         return  1 # id(self)
#
#
#
#



# ** interval algebra for parsing**


# class Interval:
#     def __init__(self, start=0, end=0):
#         """
#         Interval (start, end)
#         @param start: int
#         @param end: int
#         """
#         if start <= end:
#             self.start = start
#             self.end = end
#         else:
#             print("Interval warning: interval ({}, {}) undefined. creating ({}, {}) instead.".format(
#                 start, end, end, start))
#             self.start = end
#             self.end = start
#
#     def __len__(self):
#         return self.end - self.start
#
#     def __repr__(self):
#         return "({},{})".format(self.start, self.end)
#
#     def __lt__(self, other):
#         return self.end <= other.start
#
#     def __gt__(self, other):
#         return self.start >= other.end
#
#     def __add__(self, other):
#         if self.start == other.end:
#             return Interval(other.start, self.end)
#         elif self.end == other.start:
#             return Interval(self.start, other.end)
#         else:
#             return None
#
#     def add_incl_empty(self, other):
#         if other is None:
#             return Interval(self.start, self.end)
#         else:
#             return self.__add__(other)
#
#
#     def distance(self, other):
#         if self.__lt__(other):
#             return other.start - self.end
#         elif self.__gt__(other):
#             return self.start - other.end
#         else:
#             print("Interval.distance warning: intervals {} and {} overlap".format(self, other))
#             return 0
#
#
# class ParseItem:
#     def __init__(self, string: str, interval: Interval):
#         """
#         for neural parsing, we keep the word for the NN and the interval to tell us what's legal
#         @param string: str of constituent
#         @param interval: Interval : span of constituent, counting between the words eg 0 the 1 cat 2 slept 3
#         """
#         self.string = string
#         self.interval = interval
#
#     def __repr__(self):
#         inter = self.interval
#         if self.interval is None:
#             inter = "(-,-)"
#         return "{}:{}".format(self.string, inter)
#
#     def combine(self, neighbour):
#         """
#         combines in whichever order works accourding to the intervals; otherwise returns None
#         @param neighbour: ParseItem
#         @return: ParseItem with strings concatenated and intervals concatenated
#         """
#
#         if neighbour.interval is None:  # ignore stuff with no interval
#             return self
#         elif self.interval is None:
#             return neighbour
#         elif neighbour.interval.end == self.interval.start:
#             return ParseItem(" ".join([neighbour.string, self.string]),
#                              Interval(neighbour.interval.start, self.interval.end))
#         elif neighbour.interval.start == self.interval.end:
#             return ParseItem(" ".join([self.string, neighbour.string]),
#                              Interval(self.interval.start, neighbour.interval.end))
#         else:
#             return None
#
#     def adjacent(self, other):
#         """
#         true if other is adjacent to self, or if either interval is None
#         @param other:
#         @return:
#         """
#
#         return other.interval is None or self.interval is None or \
#                other.interval.end == self.interval.start or other.interval.start == self.interval.end
#
#
# def combine_parse_items(ps):
#     """
#     function for the algebra; takes list of 2 ParseItems and combines them
#     @param ps: ParseItem list of len 2
#     @return: ParseItem
#     """
#     assert len(ps) == 2
#     return ps[0].combine(ps[1])
#
#
# def silent():
#     """
#     No interval means we're adjacent to everything
#     @return:
#     """
#     return ParseItem("", None)
#
#
# add_silent = AlgebraOp("null", lambda l: l[0])
#
#
# # ** ALGEBRA **
# parse_item_algebra = Algebra(ops={"combine": AlgebraOp("combine", combine_parse_items),
#                                   "null_right": add_silent,
#                                   "null_left": add_silent,
#                                   "epsilon": AlgebraOp("epsilon", silent())
#                                   }, name="Parse Item Algebra")
# parse_item_algebra.add_constant_maker()
#
#
#


class Action:
    def __init__(self, name, function, f1=None, f2=None, index=None):
        """
        actions are mg deduction rules with their parameters
        @param name: str : a name for printing, eg reduce
        @param function: a deduction rule of a mg
        @param f1: first or only feature parameter
        @param f2: second feature just for move_2, for store
        @param index: index of stack item for reduce_2
        """
        self.name = name
        self.function = function
        self.f1 = f1
        self.f2 = f2
        self.i = index

    def __repr__(self):
        return " ".join([str(prop) for prop in [self.name, self.i, self.f1, self.f2] if prop is not None])


class Parser:
    def __init__(self, sentence, features=[], inner_algebra=parse_item_algebra):
        """
        Shift-reduce mg, inspired by RNNGs and Milos
        initialises with a sentence, empty stack, action list
        minimalist algebra is assumed to have a structure similar to the one generated by basic_mg_ops,
         so we refer to mg ops by key
        @param sentence : list
        """

        self.buffer = sentence  # list of StringWithInterval representing sentence
        self.stack = []         # list of Expr
        self.inner_algebra = inner_algebra    # the inner algebra of the minimalist algebra, needs to have Intervals
        self.actions = []       # Action list: keeps track of the actions we've taken
        # this is the same as in a MinimalistAlgebra
        self.constant_maker = lambda word: Expr(self.inner_algebra.constant_maker(word).function)
        self.merge_properties = MGAlgebraOpProperties({"main": inner_algebra.ops["combine"]})
        self.features = features
        self.shift_action = Action("shift", self.shift)
        self.reduce_1_action = Action("reduce_1", self.reduce_1)

    # *** DEDUCTION RULES ***

    def shift(self):
        """
        removes first word of buffer and moves to end of stack
        """
        # check if it's legal
        if len(self.buffer) == 0:
            print("can't shift: buffer is empty")
            exit()
        # get the item
        top = self.buffer.pop(0)
        # turn it into an Expression and append it to the stack
        self.stack.append(self.constant_maker(top))

    def reduce_1(self):
        """
        Applies Compose function to the last two elements on the stack, in whatever order the intervals say
        Algebra doesn't distinguish between left selecting right or right selecting left, so we don't either
        """
        self.check_reduce()  # check if it's legal
        # get the 2 items
        top = self.stack.pop()
        second = self.stack[-1]

        # check they're adjacent
        if not top.structure.successor(second.structure):
            print("reduce_1 error: {} and {} not adjacent".format(top, second))
            exit()

        # # check types
        # if type(top) != Expr or type(second) != Expr:
        #     print("Reduce_1 error: trying to merge types {} and {}".format(type(top), type(second)))

        # merge second last with last
        second.merge_1(self.merge_properties, top)


    def reduce_2_top_selects_i(self, j, f):
        """
        merges the last and jth elements of the stack, storing the ith
        @param j : int : select the jth item in the stack
        @param f : string of feature where we'll store the mover
        """
        self.check_reduce()  # check if it's legal
        self.check_reduce_2(j)

        # get the 2 items
        # ith = self.stack[i]
        # remove the ith because it's going into storage
        ith = self.stack.pop(j)
        top = self.stack[-1]

        # # check types
        # if type(ith) != Expr or type(top) != Expr:
        #     print("Reduce_1 error: trying to merge types {} and {}".format(type(top), type(ith)))

        op_properties = MGAlgebraOpProperties({"main": self.inner_algebra.ops["null_right"], "store": f})

        # # replace mover with feature marker
        # self.stack[i] = Feature(f)

        # merge and store in feature slot
        top.merge_2(op_properties, ith)

    def reduce_2_i_selects_top(self, j, f):
        """
        merges the last and ith elements of the stack, storing the top
        @param j : int : select the ith item in the stack
        @param f : string: feature where we'll store the mover
        """
        self.check_reduce()  # check if it's legal
        self.check_reduce_2(j)

        # get the 2 items
        ith = self.stack[j]
        top = self.stack.pop()
        #top = self.stack[-1]

        # # check types
        # if type(ith) != Expr or type(top) != Expr:
        #     print("Reduce_1 error: trying to merge types {} and {}".format(type(top), type(ith)))

        op_properties = MGAlgebraOpProperties({"main": self.inner_algebra.ops["null_right"], "store": f})

        # # replace mover with feature marker
        # self.stack[top] = Feature(f)

        # merge and store in feature slot
        ith.merge_2(op_properties, top)

    def store_top(self, f):
        """
        Equivalent of merging an epsilon and moving its sister. Puts the top stack item into its own storage
        @param f: feature where to store top stack item
        @return:
        """
        op_properties = MGAlgebraOpProperties({"main": self.inner_algebra.ops["null_right"], "store": f})
        top = self.stack.pop()
        empty_expression = Expr(self.inner_algebra.ops["epsilon"].function, {})
        empty_expression.merge_2(op_properties, top)
        self.stack.append(empty_expression)

    def move(self, f):
        """
        Move in top to left or right as appropriate according to intervals
        @return:
        """
        # f = self.stack.pop(-2)
        # if type(f) == Feature:

        top = self.stack[-1]

        if not top.structure.successor(top.movers[f]):
            print("move error: {} and {} not adjacent".format(top, top.movers[f]))


        # just make an operation that moves to the left and retrieves from the given slot
        op_properties = MGAlgebraOpProperties({"main": self.inner_algebra.ops["combine"], "retrieve": f})
        top.move_1(op_properties)

    # else:
        #     print("Move error: no feature marker")
        #     exit()


    def move_2(self, f1, f2):
        """
        Applies Move 2, and changes f1 tag to f2
        @param f2: Feature
        @param f1: Feature
        @return:
        """
        # if f1 in self.stack[-1].movers:
        op_properties = MGAlgebraOpProperties({"main": self.inner_algebra.ops["null_left"],
                                               "retrieve": f1, "store": f2})
        self.stack[-1].move_2(op_properties)
        # self.stack = [f2 if x == f1 else x for x in self.stack]  # replace f1 marker with f2.
        # else:
        #     print("move_2 error: no {} feature present".format(f1))

    def check_reduce(self):
        """
        Checks if reduce is possible, by checking if there are at least 2 real items on the stack
        Exits if not
        """
        # real_stack = [item for item in self.stack if type(item) == Expr]  # exclude Feature markers

        if len(self.stack) < 2:
            print("can't reduce: too few items on stack")
            exit()


    def check_reduce_2(self,i):
        """
        Checks if reduce_2 is possible, by checking if the item asked for exists, and isn't last on the stack
        Exits if not
        @param i : int
        """
        if i > len(self.stack) - 2:  # either we're asked to merge it with itself, or there's no such stack element
            print("reduce_2 error: index {} is too high".format(i))
            exit()

    def step(self, action: Action):
        """
        Applies an action and appends it to the action list
        @param action: Action
        @return:
        """
        if action == self.shift_action:
            self.shift()
        elif action == self.reduce_1_action:
            self.reduce_1()
        elif action.function == self.reduce_2_top_selects_i:
            self.reduce_2_top_selects_i(action.i, action.f1)
        elif action.function == self.reduce_2_i_selects_top:
            self.reduce_2_i_selects_top(action.i, action.f1)
        elif action.function == self.store_top:
            self.store_top(action.f1)
        elif action.function == self.move:
            self.move(action.f1)
        elif action.function == self.move_2:
            self.move_2(action.f1, action.f2)
        self.actions.append(action)

    def get_possible_actions(self):
        """
        Determines which actions are possible right now
        @return: Action list of possibilities
        """
        # make a list of possible actions
        possibles = []

        # shift
        if self.buffer:
            possibles.append(Action("shift", self.shift))

        # Merge
        if len(self.stack) > 1:
            top = self.stack[-1]
            second = self.stack[-2]

            # reduce_1
            if top.structure.successor(second.structure) and smc(top.movers, second.movers):
                possibles.append(Action("reduce_1", self.reduce_1))

            # reduce_2
            # only consider pairs that don't violate SMC
            compatible = [(j, item) for j, item in enumerate(self.stack) if smc(top.movers, item.movers)]
            # check SMC for each f
            for f in self.features:
                if f not in top.movers:
                    possibles += [Action("reduce_2_top_selects_i", self.reduce_2_top_selects_i, f1=f, index=j)
                                  for (j, item) in compatible if f not in item.movers]
                    possibles += [Action("reduce_2_i_selects_top", self.reduce_2_i_selects_top, f1=f, index=j)
                                  for (j, item) in compatible if f not in item.movers]



        if len(self.stack) > 0:
            top = self.stack[-1]

            # merge_2 with epsilon
            for f in self.features:
                if f not in top.movers:
                    possibles.append(Action("store_top", self.store_top, f1=f))

            # Move

            # move_1
            # we only move in the top item
            for f in top.movers:
                # check if they're adjacent
                if top.structure.successor(top.movers[f]):
                    possibles.append(Action("move", self.move, f1=f))


            # move_2
            # NB: we're not allowing f->f move here
            for f in top.movers:
                for g in self.features:
                    if g not in top.movers:
                        possibles.append(Action("move_2", self.move_2, f1=f, f2=g))

        return possibles


    def pretty_print(self):
        print("Buffer: ", self.buffer)
        print("Stack: ", self.stack)
        print("Actions: ", self.actions)

    def done(self):
        """
        Parsing is complete if we've used up the buffer and there's only one item on the stack, with an empty mover list
        @return boolean (true if done)
        """
        if len(self.buffer) == 0 and len(self.stack) == 1 and self.stack[0].movers == {}:
            return True
        else:
            return False

    def run(self, actions, verbose=True):
        """
         a run of the mg. note it already has its sentence
        @param verbose: If True (default), prints each step
        @param actions: Action list from the oracle
        @return: True if we got a complete structure and ran out of oracle actions at the same time, otherwise False
        """
        i = 1
        for a in actions:
            if not self.done():
                if verbose: print("\nstep ", i, ": ", a)
                i += 1
                self.step(a)
                if verbose:
                    self.pretty_print()
                    print("next possibilities: ", self.get_possible_actions())

            else:
                if verbose:
                    print("mg done before action list complete")
                    self.pretty_print()
                return False
        if verbose:
            print("\n Success!")
        return True


def index_buffer(sentence):
    """
    turns a sentence into a list of ParseItems. This is the buffer.
    @param sentence: list of str
    @return: list of ParseItem
    """
    return [ParseItem(word, Interval(j, j + 1)) for j, word in enumerate(sentence)]



def make_action_list(parser, a_list):
    """
    For running the mg with a list of actions from the oracle
    @param parser: initialised mg with sentence
    @param a_list: string list -- see cases
    @return:
    """
    actions = []
    while len(a_list) > 0:
        a = a_list.pop(0)
        if a == "shift":
            actions.append(parser.shift_action)
        elif a == "reduce_1":
            actions.append(parser.reduce_1_action)
        elif a.startswith("reduce_2_top_selects_i"):
            parts = a.split(" ")
            actions.append(Action("reduce_2_top_selects_i", parser.reduce_2_top_selects_i, index=int(parts[1]), f1=parts[2]))
        elif a.startswith("reduce_2_i_selects_top"):
            parts = a.split(" ")
            actions.append(Action("reduce_2_i_selects_top", parser.reduce_2_i_selects_top, index=int(parts[1]), f1=parts[2]))
        elif a.startswith("store_top"):
            parts = a.split(" ")
            actions.append(Action("store_top", parser.store_top, f1=parts[1]))
        elif a.startswith("move "):
            parts = a.split(" ")
            actions.append(Action("move", parser.move, f1=parts[1]))
        elif a.startswith("move_2 "):
            parts = a.split(" ")
            actions.append(Action("move_2", parser.move_2, f1=parts[1], f2=parts[2]))
        else:
            print("I don't understand the function")

    return actions


def parse(s, actions):
    """
    Wrapper function
    @param s: string list
    @param actions: string list of actions in the style in make_action_list
    @return:
    """
    p = Parser(index_buffer(s), ["k", "wh"])
    a = make_action_list(p, actions)
    return(p.run(a))


# # testing
#
# wh = Feature("wh")
# who = Expr("who")
#
# slept = Expr("slept")
#
# slept.concat_right(who)
#
# print(str(slept))
#
#
#
s = "the hungry cat slept".split(" ")
alist = ['shift', 'shift', 'shift', 'reduce_1', 'reduce_1', 'shift', 'reduce_2_top_selects_i 0 k', 'move k']

parse(s, alist)

s = "a b".split(" ")
alist = ["shift", "shift", "reduce_2_i_selects_top 0 k", "store_top wh", "move k", "move wh"]

parse(s, alist)

# p = Parser(index_buffer(s), string_with_interval_algebra, ["k", "wh"])
#
# actions = make_action_list(p, alist)
#
# print(actions)
#
# p.run(actions)

#
# s = "who slept".split(" ")
#
# p = Parser(s, concat)
#
# run(p, ["shift", "shift", "reduce_2 0 k", "move k wh", "move"])