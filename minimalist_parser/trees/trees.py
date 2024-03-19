"""
Defines Tree in general. A tree contains a parent and a list of children, which are themselves Trees
Algebra terms inherit from Trees

@author Meaghan Fowlie
"""


import nltk


class Tree:
    """
    Implementation of trees, with parents and a list of children,
     where each child is a Tree.
    Attributes:
        parent: of whatever type the nodes are. If None, empty tree
        children: Tree list. If None, this is a leaf.
    """

    def __init__(self, parent=None, children=None):
        """
        Initialise a Tree
        @param parent: whatever type you want your tree nodes to be
        @param children: Tree list
        """
        assert children is None or type(children) == list,\
            f"{children} should be a list"

        self.parent = parent
        self.children = children

    def __repr__(self):
        """
        eg NP(D(the), N(cat))
        @return: string
        """
        s = "\n" + repr(self.parent)
        if self.children is not None:
            s += f"("
            child = self.children[0]
            s += repr(child)
            for child in self.children[1:]:
                s += ", "
                s += repr(child)
            s += ")"

        return s

    def __eq__(self, other):
        return (isinstance(self, type(other)) or isinstance(other, type(self))) and self.parent == other.parent and self.children == other.children

    def is_leaf(self):
        return self.children is None

    class Variable:
        """
        A variable class for contexts so we don't accidentally use existing labels
        Attributes:
            number: int, meaning variable # (eg x_0 has number=0)
            name: string representation eg x_0
        """

        def __init__(self, number):
            """
            initialise a Variable
            @param number: int
            """
            self.number = number
            self.name = f"x_{self.number}"

        def __repr__(self):
            return self.name

    def replace_variables(self, replacements: dict):
        """
        Replaces a set of variables in a context with new subtrees
        Note that there may be variables in self that don't have variables in
         the replacements dict, and there may be keys in the replacements
          that are not in self.
        @param replacements: dict from ints to Trees.
                key: int corresponding to Variable numbers
                value: Tree to replace that variable
        @return: a Tree, which is just like context except the leaves labeled
            with variables that are in replacements are replaced by their
             corresponding Trees.
        """
        if self.children is None:
            # if you're at a leaf, check if this is variable
            if type(self.parent) == self.Variable:
                # see if replacements has the variable
                for variable in replacements:
                    # if so, replace variable with subtree
                    if self.parent.number == variable:
                        return replacements[variable]
            else:
                return self

        else:
            # recursive step: apply this function to the children
            out_trees = [self.children[i].replace_variables(
                                          replacements)
                         for i in range(len(self.children))]

            # type(self) allows this to work for subtypes of Tree
            # such as AlgebraTerm
            return type(self)(self.parent, out_trees)

    def to_nltk_tree(self):
        """
        Return an NLTK tree version of the tree
        Note this may not work if the str of the nodes makes something NLTK can't handle
        @return: nltk.Tree
        """
        if self.children is None:
            return nltk.Tree(f"{self.parent}", [])
        else:
            new_kids = [kid.to_nltk_tree() for kid in self.children]
            return nltk.Tree(f"{self.parent}", new_kids)

    @staticmethod
    def nltk_tree2tree(nltk_tree: nltk.Tree):
        """
        turns a Tree from the nltk package into a Tree as defined here
        @param nltk_tree: input tree
        @return: Tree
        """
        parent = nltk_tree.label()
        if nltk_tree.height() == 1:
            # leaf
            return Tree(parent)
        else:
            return Tree(parent, [Tree.nltk_tree2tree(kid) for kid in nltk_tree])

    def latex_forest(self):
        """
        print LaTeX code for tree in the Forest package
        @return: str
        """
        def _latex_forest(t):
            """
            makes the tree, minus the \begin{forest} and \end{forest}
            @param t:
            @return:
            """
            s = ""
            if t.children is None:
                # escape the underscores
                s += "[{" + repr(t.parent) + "}]"
            else:
                s += "[{" + repr(t.parent) + "}"
                for kid in t.children:
                    s += _latex_forest(kid)
                s += "]"
            return s

        tree_as_string = _latex_forest(self)
        special_characters = "#$%&~_^"
        for c in special_characters:
            tree_as_string = tree_as_string.replace(c, f"\\{c}")
        tree_as_string = tree_as_string.replace('<', '$<$').replace('>', '$>$')
        return "\\begin{forest}" +  tree_as_string + "\\end{forest}"

    def type_tree(self):
        """
        Creates a string in the form of a tree with just the type of each node
        @return:
        """
        s = str(type(self.parent))
        if self.children is not None:
            s += "("
            child = self.children[0].type_tree()
            s += child
            for child in self.children[1:]:
                s += ", "
                s += child.type_tree()
            s += ")"
        return s

def list2tree(t):
    """
    helper for making easier trees
    NB: any leaves labelled with int are interpreted as variables

    First element of list is the parent label, any others are children,
        so subtrees are lists inside the list.
        eg: list2tree(["f", ["ab"], ["g", [0], ["c"]]]) =
        'f'('ab', 'g'(x_0, 'c'))
    @param t: list
    @return: Tree
    """
    parent = t[0]
    if len(t) == 1:
        if type(parent) == int:
            return Tree(Tree.Variable(parent))
        else:
            return Tree(parent)
    else:
        return Tree(parent, [list2tree(kid) for kid in t[1:]])

