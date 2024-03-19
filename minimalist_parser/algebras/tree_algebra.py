from minimalist_parser.algebras.algebra import AlgebraTerm, AlgebraOp, Algebra, rev
import string_algebra as str_alg


# ***Bare Tree algebra***

# head on the right
lr = AlgebraOp("f_<",
               lambda kids:
               AlgebraTerm(AlgebraOp("<", str_alg.concat_op.function), kids))
# head on the left
rl = AlgebraOp("f_>",
               lambda kids:
               AlgebraTerm(AlgebraOp(">", str_alg.concat_op.function),
                           rev(kids)))

tree_algebra = Algebra(ops={"concat_right": lr, "concat_left": rl})
tree_algebra.spellout_function = lambda tree: tree.evaluate()

# constants are interpreted as terms too
tree_algebra.add_constant_maker(lambda leaf: AlgebraOp(leaf, AlgebraTerm(str_alg.string_alg.constant_maker(leaf))))


def extract_head(t):
    """
    given a bare tree, returns the tree with the head replaced by variable x_0
        and the head
    @param t: Tree with "<" and ">" inner nodes
    @return: pair of Trees (context with x_0 in place of head, head)
    """
    def _extract_head(full_tree, head):

        if head.parent.name == "<":
            # keep searching in left daughter
            new_kids = [_extract_head(full_tree.children[0],
                                      # only the first output of _extract_head
                                      # is the child
                                      full_tree.children[0])[0],
                        full_tree.children[1]]
            return AlgebraTerm(full_tree.parent,
                               new_kids), full_tree.children[0]
        elif head.parent.name == ">":
            # keeping searching in right child
            new_kids = [full_tree.children[0],
                        _extract_head(full_tree.children[1],
                                      full_tree.children[1])[0]]
            return AlgebraTerm(full_tree.parent,
                               new_kids), full_tree.children[1]
        else:
            # return variable x_0 for the context, and the head
            return AlgebraTerm(AlgebraTerm.Variable(0)), full_tree

    # start with the full tree as second argument,
    # gradually pare down to the head
    return _extract_head(t, t)


def join_heads(kids):
    """
    A function from AlgebraTerms to an AlgebraTerm, interpreted to
        concatenate the interpretations of the children with a hyphen
    @param kids: AlgebraTerm list
    @return: AlgebraTerm with parent "." meaning "concat with -"
    """
    return AlgebraTerm(AlgebraOp(".", str_alg.suffix.function), kids)


def hm_suf(t_0, t_1):
    """
    A preprocessing function for head movement
    move the head of t_1 up to t_0 and replace it with a trace marker
    @param t_0: AlgebraTerm of the tree algebra
    @param t_1: AlgebraTerm of the tree algebra
    @return: pair of AlgebraTerms of the tree algebra
    """
    context_0, head_0 = extract_head(t_0)
    context_1, head_1 = extract_head(t_1)
    new_head = join_heads([head_1, head_0])
    t = AlgebraTerm(str_alg.trace)
    return [context_0.replace_variables({0: new_head}),
            context_1.replace_variables({0: t})]


def al_suf(t_0, t_1):
    """
    A preprocessing function for affix lowering
    move the head of t_1 down to t_0 and replace it with a trace marker
    @param t_0: AlgebraTerm of the tree algebra
    @param t_1: AlgebraTerm of the tree algebra
    @return: pair of AlgebraTerms of the tree algebra
    """
    context_0, head_0 = extract_head(t_0)
    context_1, head_1 = extract_head(t_1)
    new_head = join_heads([head_1, head_0])
    t = AlgebraTerm(str_alg.trace)
    return [context_0.replace_variables({0: t}),
            context_1.replace_variables({0: new_head})]


if __name__ == "__main__":
    from minimalist_parser.trees import Tree

    print("Algebra Term to merge hooge, snow, and fall")
    print(
        AlgebraTerm(tree_algebra.ops["lr"],
                      [AlgebraTerm(tree_algebra.constant_maker("hooge")),
                       AlgebraTerm(tree_algebra.ops["lr"],
                                   [AlgebraTerm(tree_algebra.constant_maker("snow")),
                                    AlgebraTerm(tree_algebra.constant_maker("fall"))])]).evaluate()
                      )

    print("\ninterpret it")
    print(
        AlgebraTerm(tree_algebra.ops["lr"],
                      [AlgebraTerm(tree_algebra.constant_maker("hooge")),
                       AlgebraTerm(tree_algebra.ops["lr"],
                                   [AlgebraTerm(tree_algebra.constant_maker("snow")),
                                    AlgebraTerm(tree_algebra.constant_maker("fall"))])]).evaluate().evaluate()
                      )


    print(AlgebraTerm(tree_algebra.ops["null_left"], [AlgebraTerm(tree_algebra.constant_maker("hooge"))]).evaluate())

    print(AlgebraTerm(tree_algebra.ops["null_left"], [AlgebraTerm(tree_algebra.constant_maker("hooge"))]).evaluate().evaluate())




    lookup = {"the": "the_const", "cat": "cat_const"}

    a = Algebra(ops={}, name="with_lookup")
    a.add_constants(lookup)

    #print(a.constant_maker("cat").function)

    # import tree_algebra

    t = Tree("lr", [Tree("the"), Tree("snow")])
    #