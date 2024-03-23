from copy import copy

#from .algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from .algebras.string_algebra import BareTreeStringAlgebra
from .algebras.am_algebra_untyped import am_alg
#from minimalist_parser.algebras.algebra_objects.graphs import SGraph
from minimalist_parser.algebras.algebra_objects.triples import Triple
from .algebras.hm_triple_algebra import HMTripleAlgebra
#from .convert_mgbank.mgbank2algebra import add_addresses, add_intervals
from .minimalism.movers import DSMCMovers, Slot
#from .minimalism.prepare_packages.addressed_triple_prepare_package import HMAddressedTriplesPreparePackages
#from .minimalism.prepare_packages.interval_prepare_package import IntervalPairPrepare
from .minimalism.prepare_packages.triple_prepare_package import HMTriplesPreparePackages
from .convert_mgbank.slots import Abar, A, R, Self, E
from minimalist_parser.algebras.algebra_objects.amr_s_graphs import AMRSGraph
from .algebras.am_algebra_untyped import make_predicate
from .minimalism.minimalist_algebra_synchronous import SynchronousTerm, MinimalistAlgebraSynchronous, \
    MinimalistFunctionSynchronous, InnerAlgebraInstructions
from .algebras.algebra import AlgebraOp
from minimalist_parser.minimalism.prepare_packages.prepare_packages_bare_trees import PreparePackagesBareTrees



# String algebra with Bare Tree terms
string_alg = BareTreeStringAlgebra(name="string_algebra")
string_alg.add_constant_maker()

# initialise triple algebra
triple_alg = HMTripleAlgebra(name="string triples")
triple_alg.add_constant_maker()
triple_prepare_packages = HMTriplesPreparePackages(inner_algebra=triple_alg)
print("***************** EMPTY ********************")
print(triple_alg.empty_leaf_operation.function)
print("done")
# addresses_alg = HMTripleAlgebra(name="address triples", component_type=list)

# initialise MG and Sync MG over triples only
# mg = MinimalistAlgebra(triple_alg, prepare_packages=triple_prepare_packages)
# smg = MinimalistAlgebraSynchronous([triple_alg],
#                                    prepare_packages=[triple_prepare_packages],
#                                    mover_type=DSMCMovers)

# # add some consonants
# cat = mg.make_leaf("cat")
# # print("###################### Trying to make constants")
# # print(cat)
# # print(cat.evaluate())
# # print(cat.evaluate().spellout())
# slept = mg.make_leaf("slept")
# # print(slept.evaluate())
# # print(slept.evaluate().inner_term)
# # print(slept.evaluate().inner_term.evaluate())
# # print("############# DONE ##############")
# merge = MinimalistFunction(mg.merge1, "merge", inner_op=triple_alg.concat_right)
# cat_slept = AlgebraTerm(merge, [slept, cat])
# # print(cat_slept.evaluate())
#
# # turn term into a SynchronousTerm
# tree_sync = smg.synchronous_term_from_minimalist_term(cat_slept, inner_algebra=triple_alg)
# # print(tree_sync)
# print("################# Interp")
# print(tree_sync.interp(triple_alg))
# print("DONE\n")
# AM algebra update
and_g = AMRSGraph({0, 1, 2}, edges={0: [(1, "op1"), (2, "op2")]}, root=0, sources={"OP1": 1, "OP2": 2},
                  node_labels={0: "and"})
and_g_op = AlgebraOp("and_s", and_g)

dream_g = make_predicate(label="dream-01", arg_numbers=[0])
am_alg.add_constants({"and_s": and_g_op,
                      "dreamt": AlgebraOp("dreamt", dream_g)}, default=am_alg.default_constant_maker)

#
# mg_am = MinimalistAlgebra(am_alg)
# print(am_alg.constants)
# cat_g = mg_am.make_leaf("cat")
# slept_g = mg_am.make_leaf("slept_mg_name")
# print(slept_g.parent.function.inner_term.evaluate())
# merge_app_s = MinimalistFunction(mg.merge1, "merge", inner_op=am_alg.ops["App_S"])
#
# cat_slept_g = AlgebraTerm(merge_app_s, [slept_g, cat_g])
# print(cat_slept_g.evaluate().spellout())
#
# tree_sync.add_algebra(am_alg, cat_slept_g)
# print("***spellout\n", tree_sync.interp(am_alg).spellout())
# print("***spellout\n", tree_sync.interp(triple_alg).spellout())
#
# print("\n*** testing evaluate")
# print(tree_sync.evaluate().spellout())
#
# print("#############")
# print(am_alg.constants["cat_mg_name"].function)
# print(am_alg.constants["slept_mg_name"].function)
# print("###########")

# example with 4 interpretations
mg = MinimalistAlgebraSynchronous([string_alg, am_alg, triple_alg],
                                  prepare_packages=[PreparePackagesBareTrees(name="for strings",
                                                                             inner_algebra=string_alg),
                                                    None, triple_prepare_packages], mover_type=DSMCMovers)
# print(mg)
# #
# print("AM constants")
# for c in am_alg.constants:
#     print(c)
#     assert isinstance(am_alg.constants[c].function,
#                       SGraph), f"{c} is not an SGraph but a {type(am_alg.constants[c].function)}"

# the = mg.constant_maker("the", string_alg)
# # print(type(the.function.inner_term.evaluate()))
# cat_s = mg.constant_maker("cat", string_alg)
# cat_g = mg.constant_maker("cat", am_alg)
# print(cat_s.function)
# print(cat_g.function.inner_term.evaluate())
# print(am_alg.constants["mary"].function)

# optionally, we give labels for the default constant maker to use if words aren't found in the constant dict
# note that we look up the words using the name of the MinimalistFunction, below
cat_functions = {
    am_alg: InnerAlgebraInstructions('poes'),
}

to_functions = {am_alg: None,  # this means there's no interpretation, so we'll skip it
                string_alg: InnerAlgebraInstructions("to_control", leaf_object="to"),
                triple_alg: InnerAlgebraInstructions("to_control", leaf_object=Triple("to")),
                }

past_functions = {string_alg: InnerAlgebraInstructions('[past]', leaf_object=string_alg.empty_leaf_operation.function),
                  triple_alg: InnerAlgebraInstructions('[past]', leaf_object=triple_alg.empty_leaf_operation.function),
                  am_alg: None,  # this means there's no interpretation, so we'll skip it
                  }

q_functions = {string_alg: InnerAlgebraInstructions('[Q]', leaf_object=str()),
               triple_alg: InnerAlgebraInstructions('[Q]', leaf_object=Triple()),
               am_alg: None,  # this means there's no interpretation, so we'll skip it
               }

did_functions = {
    am_alg: None,  # this means there's no interpretation, so we'll skip it
}

# should be +conj
and_functions = {am_alg: InnerAlgebraInstructions("and_s")}


# leaves
cat = mg.make_leaf("cat", cat_functions)
slept = mg.make_leaf("slept")
dreamt = mg.make_leaf("dreamt")
tried = mg.make_leaf("tried")
to = mg.make_leaf("to_control", to_functions)
and_term = mg.make_leaf("and", and_functions, conj=True)
past = mg.make_leaf("[past]", past_functions)
did = mg.make_leaf("did", did_functions)
q = mg.make_leaf("[Q]", q_functions)
sleep = mg.make_leaf("sleep")
dream = mg.make_leaf("dream")
the = mg.make_leaf("the", {am_alg: None})
dog = mg.make_leaf("dog")

#
# print("############### DID ###################")
# print(did_f)
# print(did_f.inner_ops[triple_alg])

# print("\n\n########### AND Term ################")
# print(and_term.evaluate().mg_type.conj)
# print(and_term.interp(am_alg))
# print(and_term.interp(am_alg).spellout())

# print("t", t)
#
# t_string_alg = t.evaluate(string_alg)
# print("t_string_alg", t_string_alg)
# interp_t_string_alg = t_string_alg.evaluate()
# print("evaluate", interp_t_string_alg)
# inner_term = interp_t_string_alg.inner_term
# print("inner", inner_term)
# string_output = inner_term.evaluate()
# print("output", string_output)
#
# print("***************")
# t_am_alg = t.evaluate(am_alg)
# print("t_am_alg", t_am_alg)
# # interp_t_am_alg = t_am_alg.evaluate()
# # print("evaluate", interp_t_am_alg)
# # inner_term = interp_t_am_alg.inner_term
# # print("inner", inner_term)
# # g_output = inner_term.evaluate()
# # print("output", g_output, type(g_output))
# print("******************")
#
# print("slept term", t2.evaluate(am_alg))
# print("***************")

# each algebra is mapped to (inner_op, prepare, reverse)
inners_s = {
    string_alg: InnerAlgebraInstructions("concat_left", reverse=True),
    am_alg: InnerAlgebraInstructions("App_S"),
}

inners_o = {
    string_alg: InnerAlgebraInstructions("concat_right"),
    am_alg: InnerAlgebraInstructions("App_O"),
}

inners_op1 = {
    string_alg: InnerAlgebraInstructions("concat_left", reverse=True),
    am_alg: InnerAlgebraInstructions("App_OP1"),
}

inners_op2 = {
    string_alg: InnerAlgebraInstructions("concat_right"),
    am_alg: InnerAlgebraInstructions("App_OP2"),
}

inners_atb_op1 = {
    string_alg: InnerAlgebraInstructions("concat_left", "excorporation", reverse=True),
    am_alg: InnerAlgebraInstructions("App_OP1"),
}

inners_atb_op2 = {
    string_alg: InnerAlgebraInstructions("concat_right", "excorporation"),
    am_alg: InnerAlgebraInstructions("App_OP2"),
}

inners_r = {
    string_alg: InnerAlgebraInstructions("concat_right"),
}

inners_hm = {
    string_alg: InnerAlgebraInstructions("concat_right", "prefix"),
}

inners_atb = {
    string_alg: InnerAlgebraInstructions("concat_right", "hm_atb"),
    am_alg: InnerAlgebraInstructions("App_OP1")
}

# add triple algebra
for ops in [inners_s, inners_r, inners_hm, inners_atb, inners_atb_op2, inners_atb_op1, inners_op1, inners_op2,
            inners_o]:
    ops[triple_alg] = copy(ops[string_alg])
    ops[triple_alg].reverse = False

merge_s = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_s,
                                        name="merge_S")
merge_o = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_o,
                                        name="merge_O")
merge_r = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_r,
                                        name="merge_right")
merge_op1 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                          inner_ops=inners_op1,
                                          name="merge_op1")
merge_op2 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                          inner_ops=inners_op2,
                                          name="merge_op2")
merge2 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge2,
                                       to_slot=Slot(A),
                                       name="merge_A")
move1 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.move1,
                                      from_slot=A,
                                      inner_ops=inners_s,
                                      name="move_A")
merge_hm = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_hm,
                                         name="merge_hm")

merge_atb = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                          inner_ops=inners_atb,
                                          name="merge_atb")

merge_atb_op1 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                              inner_ops=inners_atb_op1,
                                              name="merge_atb_op1")
merge_atb_op2 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                              inner_ops=inners_atb_op2,
                                              name="merge_atb_op2")

# did the cat sleep and the dog dream
# has ATB head movement
tree_atb_hm = SynchronousTerm(merge_hm, [
    q,
    SynchronousTerm(merge_atb, [
        SynchronousTerm(merge_atb_op2, [
            and_term,
            SynchronousTerm(move1, [
                SynchronousTerm(merge_r, [
                    did,
                    SynchronousTerm(merge2, [
                        sleep,
                        SynchronousTerm(merge_r,
                                        [
                                            the,
                                            cat
                                        ])
                    ])
                ])
            ])
        ]),
        SynchronousTerm(move1, [
            SynchronousTerm(merge_r, [
                did,
                SynchronousTerm(merge2, [
                    dream,
                    SynchronousTerm(merge_r,
                                    [
                                        the,
                                        dog
                                    ])
                ])
            ])
        ])
    ])
])

# optionally, we give labels for the default constant maker to use if words aren't found in the constant dict
# note that we look up the words using the name of the MinimalistFunction, below
cat_functions = {
    am_alg: InnerAlgebraInstructions('poes'),
}

to_functions = {am_alg: None,  # this means there's no interpretation, so we'll skip it
                string_alg: InnerAlgebraInstructions("to_control", leaf_object="to"),
                triple_alg: InnerAlgebraInstructions("to_control", leaf_object=Triple("to")),
                }

past_functions = {string_alg: InnerAlgebraInstructions('[past]', leaf_object=string_alg.empty_leaf_operation.function),
                  triple_alg: InnerAlgebraInstructions('[past]', leaf_object=triple_alg.empty_leaf_operation.function),
                  am_alg: None,  # this means there's no interpretation, so we'll skip it
                  }

q_functions = {string_alg: InnerAlgebraInstructions('[Q]', leaf_object=str()),
               triple_alg: InnerAlgebraInstructions('[Q]', leaf_object=Triple()),
               am_alg: None,  # this means there's no interpretation, so we'll skip it
               }

did_functions = {
    am_alg: None,  # this means there's no interpretation, so we'll skip it
}

# should be +conj
and_functions = {am_alg: InnerAlgebraInstructions("and_s")}


# leaves
cat = mg.make_leaf("cat", cat_functions)
slept = mg.make_leaf("slept")
dreamt = mg.make_leaf("dreamt")
tried = mg.make_leaf("tried")
to = mg.make_leaf("to_control", to_functions)
and_term = mg.make_leaf("and", and_functions, conj=True)
past = mg.make_leaf("[past]", past_functions)
did = mg.make_leaf("did", did_functions)
q = mg.make_leaf("[Q]", q_functions)
sleep = mg.make_leaf("sleep")
dream = mg.make_leaf("dream")
the = mg.make_leaf("the", {am_alg: None})
dog = mg.make_leaf("dog")



# the cat slept and dreamt
# has ATB phrasal movement
tree_atb = SynchronousTerm(move1,
                           [
                               SynchronousTerm(merge_r,
                                               [
                                                   past,
                                                   SynchronousTerm(merge_op1,
                                                                   [
                                                                       SynchronousTerm(merge_op2, [
                                                                           and_term,
                                                                           SynchronousTerm(merge2, [
                                                                               slept,
                                                                               SynchronousTerm(merge_r,
                                                                                               [
                                                                                                   the,
                                                                                                   cat
                                                                                               ])
                                                                           ])
                                                                       ]),
                                                                       SynchronousTerm(merge2, [
                                                                           dreamt,
                                                                           SynchronousTerm(merge_r,
                                                                                           [
                                                                                               the,
                                                                                               cat
                                                                                           ])
                                                                       ])
                                                                   ]

                                                                   )
                                               ])
                           ])

# optionally, we give labels for the default constant maker to use if words aren't found in the constant dict
# note that we look up the words using the name of the MinimalistFunction, below
cat_functions = {
    am_alg: InnerAlgebraInstructions('poes'),
}

to_functions = {am_alg: None,  # this means there's no interpretation, so we'll skip it
                string_alg: InnerAlgebraInstructions("to_control", leaf_object="to"),
                triple_alg: InnerAlgebraInstructions("to_control", leaf_object=Triple("to")),
                }

past_functions = {string_alg: InnerAlgebraInstructions('[past]', leaf_object=string_alg.empty_leaf_operation.function),
                  triple_alg: InnerAlgebraInstructions('[past]', leaf_object=triple_alg.empty_leaf_operation.function),
                  am_alg: None,  # this means there's no interpretation, so we'll skip it
                  }

q_functions = {string_alg: InnerAlgebraInstructions('[Q]', leaf_object=str()),
               triple_alg: InnerAlgebraInstructions('[Q]', leaf_object=Triple()),
               am_alg: None,  # this means there's no interpretation, so we'll skip it
               }

did_functions = {
    am_alg: None,  # this means there's no interpretation, so we'll skip it
}

# should be +conj
and_functions = {am_alg: InnerAlgebraInstructions("and_s")}


# leaves
cat = mg.make_leaf("cat", cat_functions)
slept = mg.make_leaf("slept")
dreamt = mg.make_leaf("dreamt")
tried = mg.make_leaf("tried")
to = mg.make_leaf("to_control", to_functions)
and_term = mg.make_leaf("and", and_functions, conj=True)
past = mg.make_leaf("[past]", past_functions)
did = mg.make_leaf("did", did_functions)
q = mg.make_leaf("[Q]", q_functions)
sleep = mg.make_leaf("sleep")
dream = mg.make_leaf("dream")
the = mg.make_leaf("the", {am_alg: None})
dog = mg.make_leaf("dog")



# term for "the cat tried to sleep"
t3 = SynchronousTerm(move1,
                     [SynchronousTerm(
                         merge_o,
                         [tried,
                          SynchronousTerm(
                              merge_r,
                              [to,
                               SynchronousTerm(
                                   merge2,
                                   [sleep,
                                    SynchronousTerm(
                                        merge_r,
                                        [the, cat]
                                    )
                                    ]
                               )
                               ]
                          )
                          ]
                     )
                     ]
                     )

cat_tried_sleep = t3

# example_interval_algebra = HMIntervalPairsAlgebra()
# interval_prepare = IntervalPairPrepare()
# example_interval_algebra.add_constant_maker()
#
# address_alg = HMTripleAlgebra("adding addresses algebra", component_type=list, addresses=True)
# address_prepare = HMAddressedTriplesPreparePackages()
#
# mg.inner_algebras[address_alg] = address_prepare
#
# for t in [t3, tree_atb, tree_atb_hm]:
#     add_addresses(t, triple_alg, address_alg)
#     address_output = t.spellout(address_alg)
#     add_intervals(t, address_output, example_interval_algebra, address_alg)
