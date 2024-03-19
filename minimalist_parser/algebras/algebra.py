"""
Algebra functionality parent class

@author Meaghan Fowlie
"""

from .. import logging
from copy import copy
from typing import Any

import nltk

from ..trees.trees import Tree

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
log = logger.debug


class AlgebraError(Exception):
    """Exception raised for errors in algebra
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = message


# **some helper functions **


def identity(x):
    """
    dummy function
    @param x:
    @return: x
    """
    return x


def rev(lijstje):
    """
    returns reversed list
    @param lijstje: list
    @return: list
    """
    lijstje.reverse()
    return lijstje


# ** main classes **

class AlgebraOp:
    """
    Algebra operation, with its syntax (name) and semantics (function)

    Attributes:
        name: str, the syntactic symbol
        function: a function from the semantics of its daughters
         to the algebra domain
    """

    def __init__(self, name, function: Any):
        """
        symbol and interpretation
        @param name: symbol
        @param function: function from daughters to domain
        """
        # try:
        assert name is None or type(name) == str, f"{name} should be a string but is a {type(name)}"
        # except AssertionError as e:
        #     print(e)
        self.name = name
        if function is None:
            function = identity
        self.function = function
        # self.__name__ = self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        ret = isinstance(other, type(self)) and self.properties() == other.properties()
        logger.debug(f"comparing {self} and {other}: {ret}")
        if not ret:
            logger.debug(f"name: {self.name} vs {other.name} ({self.name == other.name})")
            logger.debug(f"function: {self.function} vs {other.function} ({self.function == other.function})")

        return ret

    def __hash__(self):
        return hash(self.properties())

    def properties(self):
        try:
            function = self.function.__name__
        except AttributeError:
            function = self.function
        return f"name: {self.name}, function: {function}"


class AlgebraTerm(Tree):
    """
    a Tree over AlgebraOps. Can be evaluated with evaluate method to get an
     element of the algebra domain.

    Attributes:
        parent: AlgebraOp
        children: list of AlgebraOp. If None, this is a leaf.
    """

    def __init__(self, parent: AlgebraOp, children=None):
        """
        A tree with an interpretation function. Depends on parent being an AlgebraOp
        @param parent: AlgebraOp
        @param children: list of AlgebraTerm. Default None, for leaves
        """
        assert isinstance(parent, AlgebraOp), f"parent should be AlgebraOp but {parent} has type {type(parent)}"
        super().__init__(parent, children)
        if children is not None:
            for child in children:
                assert isinstance(child, AlgebraTerm), f"child should be AlgebraTerm but is {type(child).__name__}"

    def __repr__(self):
        try:
            ret = repr(self.to_nltk_tree())
            ret = ret.replace("Tree", "")
            ret = ret.replace(", []", "")
        except:
            ret = repr(self.parent) + "["
            for child in self.children:
                ret += repr(child) + ",\n"
            ret += "]"
        return ret

    def __hash__(self):
        return hash(repr(self))

    @staticmethod
    def print_differences(tree_1, tree_2):
        """
        Recursively print the differences between two trees
        @param tree_1: Tree
        @param tree_2: Tree
        """
        if tree_1.parent != tree_2.parent:
            print(tree_1.parent.properties(), "VS", tree_2.parent.properties())
        if tree_1.children is not None and tree_2.children is not None:
            for kid_1, kid_2 in zip(tree_1.children, tree_2.children):
                tree_1.print_differences(kid_1, kid_2)

    class Variable(AlgebraOp):
        """
        A variable class for contexts so we don't accidentally use existing labels
        """

        def __init__(self, number):
            super().__init__(f"x_{number}", lambda x: x)
            self.number = number

    def is_leaf(self):
        return self.children is None

    def evaluate(self):
        """
        Recursively interpret the term
        @return: object of the domain of the algebra
        """
        logger.debug(f"Interpreting term {self}")
        assert issubclass(type(self.parent),
                          AlgebraOp), f"Should be an AlgebraOp, but instead is a {type(self.parent).__name__}"
        if self.is_leaf():
            logger.debug(f"interpreting leaf: {self.parent.function} of type {type(self.parent.function).__name__}")

            return self.parent.function
        else:
            try:
                logger.debug(f"evaluating {self.parent} of type {type(self.parent).__name__} with {len(self.children)} children")
                logger.debug(f"children have types {[type(kid).__name__ for kid in self.children]}")
            except TypeError as e:
                logger.warning(e)
                logger.warning(f"\n***********parent:{self.parent.name}")
                logger.warning(f"{type(self.parent).__name__}")
                logger.warning(f"function {self.parent.function}")
                logger.warning(f"name {self.parent.name}")
                logger.warning("***************\n")
            except Exception as e:
                logger.exception("what??")
                raise
            logger.debug(f"Applying {self.parent} to {self.children} of types {[type(kid) for kid in self.children]}")
            if not callable(self.parent.function):
                raise AlgebraError(f"Internal node interpretation of type"
                                   f" {type(self.parent.function).__name__}"
                                   f" is not a callable function")
            try:
                # recursive step
                logger.debug(f"kids: {self.children}")
                output = self.parent.function([kid.evaluate() for kid in self.children])
                logger.debug(f"building {output} of type {type(output).__name__}")
                return output
            except IndexError:
                raise AlgebraError(
                    f"{self.parent} requires more arguments than provided ({[kid.parent for kid in self.children]})")
            except TypeError:
                logger.exception(f"error with parent function {self.parent.function}")
                raise

    def annotated_tree(self):
        """
        Get a Tree annotated at each node with the result of interpreting the subtree it dominates
        @return: Tree
        """
        if self.is_leaf():
            return Tree(self.evaluate())
        else:
            output = self.parent.function([kid.evaluate() for kid in self.children])
            return Tree(output, [kid.annotated_tree() for kid in self.children])

    def type_tree(self):
        """
        Creates a string in the form of a tree with just the type of each node
        @return:
        """
        s = str(type(self.parent.function))
        if self.children is not None:
            s += "("
            child = self.children[0].type_tree()
            s += child
            for child in self.children[1:]:
                s += ", "
                s += child.type_tree()
            s += ")"
        return s

    def function_tree(self):
        """
        Creates a string in the form of a tree with just the type of each node
        @return:
        """
        if self.is_leaf():
            return Tree(self.parent.function)
        else:

            return Tree(self.parent, [kid.function_tree() for kid in self.children])

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
                s += "[" + t.parent.name.replace("[", "{[").replace("]", "]}") + "]"
            else:
                s += "[" + t.parent.name.replace("[", "{[").replace("]", "]}")
                for kid in t.children:
                    s += _latex_forest(kid)
                s += "]"
            return s

        tree_as_string = _latex_forest(self)
        special_characters = "#$%&~_^"
        for c in special_characters:
            tree_as_string = tree_as_string.replace(c, f"\\{c}")
        return "\\begin{forest}" + tree_as_string + "\\end{forest}"

    def to_nltk_tree(self):
        """
        Return an NLTK tree version of the term
        @return: nltk.Tree
        """
        if self.children is None:
            return nltk.Tree(self.parent.name, [])
        else:
            for kid in self.children:
                assert kid is not None, f"A child in {self.children} is None!"
            new_kids = [kid.to_nltk_tree() for kid in self.children]
            return nltk.Tree(self.parent.name, new_kids)


class Algebra:
    """
    A algebra with AlgebraOps stored in a dict from their names to the operations
    May include a constant_maker, which automatically generates zero-ary
     functions from inputs such as strings.
    Constants can also be added as a dict, and then the constant_maker just
     looks things up in self.constants.

    Attributes:
        domain_type: Type of the objects of the algebra. Default str.
        ops: dict from strings to AlgebraOp. Usually just for non-constants.
        name: str. Name for the algebra mostly for printing. Default "algebra".
        empty_leaf: AlgebraTerm of the empty_object.
        constant_maker: function from some input, usually str, to AlgebraOps
         whose "function" is an element of the domain of the algebra.
         Default None
        spellout_function: function from objects to strings (optional)
    """

    def __init__(self, domain_type=str,
                 ops=None,
                 name="algebra",
                 constant_maker=None,
                 spellout=None,
                 zero=None,
                 meta=None):
        """
        makes an algebra with a dict of (usually non-zero-ary) AlgebraOps,
         and optionally a name and a way to make a constant out of any string.
          You can also add a dict of constants using add_constants and look them up.
        @param domain_type: Type of the obejcts of the algebra. Default str.
        @param ops: str to AlgebraOp dict
        @param name: string (opt)
        @param constant_maker: function from string to zero-ary AlgebraOp (opt)
        @param spellout: function from algebra objects to strings (optional)
        @param zero: AlgebraOp for empty item (optional;
         default makes an empty AlgebraOp("[empty]", domain_type()) if possible with this domain type.
         Otherwise, None.
        """
        self.domain_type = domain_type
        assert isinstance(zero,
                          AlgebraOp) or zero is None, f"{zero} should be an AlgebraOp but is a {type(zero).__name__}"
        if zero is None:
            try:
                self.empty_leaf_operation = AlgebraOp("[empty]", domain_type())
            except:
                logger.warning("No empty leaf operation possible")
                self.empty_leaf_operation = None
        else:
            self.empty_leaf_operation = zero

        self.name = name
        self.constant_maker = constant_maker
        if ops is None:
            self.ops = {}
        else:
            self.ops = ops

        # can add a dict of constants instead of a generator.
        # See add_constants below
        self.constants = None

        self.spellout_function = spellout if spellout is not None else lambda x: x
        # meta data about the algebra
        self.meta = {} if meta is None else meta

        try:
            self.empty_leaf = AlgebraTerm(self.empty_leaf_operation)
        except Exception as e:
            logger.warning(f"couldn't make empty leaf: {e}")
            self.empty_leaf = None
            # raise

    def __repr__(self):
        return str(self)
        # s = f"{self.name}\n  ops: {self.ops_repr()}\n" #constant maker: {self.constant_maker}\n"
        # if self.constant_maker:
        #     s += self.constant_maker_repr()
        #
        # return s

    def __str__(self):
        return self.name

    def get_op_by_name(self, name):
        """
        Try to turn a string into an AlgebraOp.
            1. Look it up in self.ops.
            2. Try to find it as the name of a function of the algebra.
            3. Try to turn it into a constant with self.constant_maker.
            4. Try to turn it into a constant with self.domain_type(name).
            5. Raise an AlgebraError if all fails.
        @param name: str: the name of the function, such as concat_right.
        @return: AlgebraOp
        """
        if name in self.ops:
            return self.ops[name]
        else:
            logger.warning(f"no op named {name} in {self.name}.ops")
            logger.debug(f"ops: {list(self.ops.keys())}")
            try:
                function = getattr(self, name)
                return AlgebraOp(name=name, function=function)
            except AttributeError:
                raise AlgebraError(f"Couldn't find {name} function in {self.name}")
                # logger.warning(f"No function called {name}; trying to make a constant.")
                # try:
                #     return self.constant_maker(name)
                # except Exception as e:
                #     try:
                #         function = self.domain_type(name)
                #         return AlgebraOp(name=name, function=function)
                #     except Exception as e:
                #         raise AlgebraError(
                #             f"Couldn't find {name} in ops, in {self.name} as a function, or make it into a constant.")

    def constant_maker_repr(self):
        s = f"  {self.constant_maker.__name__}"
        try:
            s += f"\n    constants w -> {self.constant_maker('w').function}\n"
        except:
            try:
                s += f"\n    constants 0 -> {self.constant_maker(0).function}\n"
            except:
                return s
        return s

    def ops_repr(self):
        return repr(self.ops)

    def add_constant_maker(self, function=None):
        """
        A function to make an arbitrary zero-ary function (constant)
        @param function: function from leaf to constant function
        default names it after the default string rep of the input leaf and
         makes the "function" the leaf itself,
         which is by definition an object in the domain of the algebra
        @return:
        """
        if function is None:
            function = self.default_constant_maker
        self.constant_maker = function

    def add_constants(self, constant_dict, default=None):
        """
        if you have a dict of constants
        @param constant_dict: string to algebra op dict
        @param default: a way to make up a constant if it's not in the dict. Function from string (key) to AlgebraOp.
         default value just uses the word as the function of the AlgebraOp
        """
        if default is None:
            default = self.default_constant_maker
        assert issubclass(type(constant_dict), dict)
        if self.constants is None:
            self.constants = {}
        self.constants.update(constant_dict)

        def look_up_constant(word, label=None):
            """
            Look up the word in the constants dictionary, and if it's not there, use the default function to make one
            @param label: str: if given, look it up in the constants dictionary
            @param word: str: string to give to the default constant maker,
                                and if no label is given, look word up in the constant dictionary
            @return: AlgebraOp
            """
            if label is None:
                label = word
            c = self.constants.get(label, default(word))
            if word in self.constants:
                logger.debug(f"found {label} in dict")
                logger.debug(self.constants[label].function)
            else:
                logger.debug(f"did not find {label} in dict; made new constant")
            # copy it so we don't change the constant with ops
            return AlgebraOp(c.name, copy(c.function))

        # make the constant_maker the function that looks words up
        # in constant_dict
        self.add_constant_maker(look_up_constant)

    def make_leaf(self, name, **kwargs):
        function = kwargs.get("function")
        if function is None:
            return AlgebraTerm(self.constant_maker(name))
        else:
            return AlgebraTerm(AlgebraOp(name=name, function=function))

    def add_op(self, op: AlgebraOp):
        if op.name not in self.ops:
            self.ops[op.name] = op
        else:
            logger.warning(f"Didn't add {op} to ops because an operation of that name was already there")

    def default_constant_maker(self, word, label=None):
        logger.debug("using default constant maker")
        if label is None:
            label = str(word)
        return AlgebraOp(label, word)


def tree2term(t: Tree, algebra):
    """
    Makes an algebra term from a tree, provided the labels are algebra operation
     dict keys and algebra.constant_maker is defined
    @param t: Tree
    @param algebra: Algebra
    @return: AlgebraTerm
    """

    if algebra.constant_maker is None:
        raise Exception(f"No constant make defined in {algebra}")

    if t.children is None:
        if type(t.parent) == Tree.Variable:
            # if it's a variable, make a variable
            return AlgebraTerm(AlgebraTerm.Variable(t.parent.number))
        else:
            # if it's a leaf, use the constant_maker
            return AlgebraTerm(algebra.constant_maker(t.parent))
    else:
        # otherwise, interpret recursively
        parent = algebra.ops[t.parent]
        kids = [tree2term(child, algebra) for child in t.children]
        return AlgebraTerm(parent, kids)


if __name__ == "__main__":
    from minimalist_parser.algebras.tree_algebra import tree_algebra

    # EXAMPLES

    # since these are terms, they themselves can also be interpreted (into strings)

    print(
        AlgebraTerm(tree_algebra.ops["lr"],
                    [AlgebraTerm(tree_algebra.constant_maker("hooge")),
                     AlgebraTerm(tree_algebra.ops["lr"],
                                 [AlgebraTerm(tree_algebra.constant_maker("snow")),
                                  AlgebraTerm(tree_algebra.constant_maker("fall"))])]).evaluate()
    )

    print(
        AlgebraTerm(tree_algebra.ops["lr"],
                    [AlgebraTerm(tree_algebra.constant_maker("hooge")),
                     AlgebraTerm(tree_algebra.ops["lr"],
                                 [AlgebraTerm(tree_algebra.constant_maker("snow")),
                                  AlgebraTerm(tree_algebra.constant_maker("fall"))])]).evaluate().evaluate()
    )

    lookup = {"the": "the_const", "cat": "cat_const"}

    a = Algebra(ops={}, name="with_lookup")
    a.add_constants(lookup)

    # print(a.constant_maker("cat").function)

    # import tree_algebra

    t = Tree("lr", [Tree("the"), Tree("snow")])
    #
    # term = tree2term(t, tree_algebra)

    # print(term.evaluate())
