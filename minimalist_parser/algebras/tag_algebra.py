from minimalist_parser.algebras.algebra import *


def concatenate_strings(a, b):
    """
    only put in a space if it's not empty
    @param a: str
    @param b: str
    @return: str
    """
    real_strings = [x for x in [a, b] if x != "" and x is not None]
    if not real_strings:
        return ""
    else:
        return " ".join(real_strings)


class StringPair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return "({}, {})".format(self.first, self.second)

    def interp(self):
        return concatenate_strings(self.first, self.second)

    def wrap_string(self, string):
        """
        Wrap around a string and return a string
        (a,b) + c = acb
        @param string: str
        @return: str
        """
        return concatenate_strings(concatenate_strings(self.first, string), self.second)

    def wrap_pair(self, pair):
        """
        Wrap around another pair
        (a,b) + (c,d) = (ac, db)
        @param pair: StringPair
        @return: StringPair
        """
        return StringPair(concatenate_strings(self.first, pair.first),
                          concatenate_strings(pair.second, self.second))

    def concat_to_rightward_string(self, string):
        """
        concatenate to string on the right
        (a,b) + c = (a,bc)
        @param string: str
        @return: StringPair
        """
        return StringPair(self.first, concatenate_strings(self.second, string))

    def concat_to_leftward_string(self, string):
        """
        concatenate to string on the left
        a + (b,c) = (ab, c)
        @param string:
        @return:
        """
        return StringPair(concatenate_strings(string, self.first), self.second)


# ab = StringPair("a", "b")
# cd = StringPair("c", "d")
#
# print(ab.wrap_string("c"))
#
# print(ab.wrap_pair(cd))
#
# print((ab.concat_to_rightward_string("c")))
#
# print(ab.concat_to_leftward_string("c"))
#

def wrap(args):
    pair, other = args[0], args[1]
    if type(other) == str:
        return pair.wrap_string(other)
    elif type(other) == StringPair:
        return pair.wrap_pair(other)
    else:
        print("wrap error: second item neither string nor StringPair")
        exit()


def concat(args):
    """
    concat x, y when args = [x,y]
    @param args: list of length two containing any combination of str and StringPair
    @return: str if a,b both str, otherwise StringPair
    """
    one, two = args[0], args[1]
    if type(one) == str and type(two) == str:  # ab
        return concatenate_strings(one, two)
    elif type(one) == str and type(two) == StringPair:  # (ab, c)
        return two.concat_to_leftward_string(one)
    elif type(one) == StringPair and type(two) == str:  # (a, bc)
        return one.concat_to_rightward_string(two)
    elif type(one) == StringPair and type(two) == StringPair:  # (ab, cd)
        return StringPair(concatenate_strings(one.first, one.second),
                          concatenate_strings(two.first, two.second))
    else:
        print("concat error: can't concat types {} and {}".format(type(one),
                                                                  type(two)))
        exit()


def concat_rev(args):
    """
    concat y,x when args = [x,y]
    @param args: list of length two containing any combination of str and StringPair
    @return: str if a,b both str, otherwise StringPair
    """
    return concat(rev(args))


def add_nothing(args):
    """
    args[0] is the thing we want; args[1] is the mover being stored in MG
    @param args: list of str and StringPair
    @return: first thing in list
    """
    return args[0]


def empty_pair():
    """
    epsilon pair
    ("","")
    @return: StringPair
    """
    return StringPair("", "")


def make_right_pair(string):
    """
    A possible way of making a constant
    a -> ("", a)
    @param string: str
    @return: StringPair
    """
    return empty_pair().concat_to_rightward_string(string)


def make_left_pair(string):
    """
    A possible way of making a constant
    a -> (a, "")
    @param string: str
    @return: StringPair
    """
    return empty_pair().concat_to_leftward_string(string)


def make_pair(string, left=False):
    if left:
        return empty_pair().concat_to_leftward_string(string)
    else:
        return empty_pair().concat_to_rightward_string(string)


def collapse(object):
    if type(object) == str:
        return object
    elif type(object) == StringPair:
        return object.evaluate()
    else:
        raise TypeError


tag_ops = {"wrap": AlgebraOp("wrap", wrap),
           "lr": AlgebraOp("concat", concat),
           "rl": AlgebraOp("concat_rev", concat_rev),
           "epsilon_pair": AlgebraOp("<e,e>", empty_pair()),
           "epsilon_string": AlgebraOp("", ""),
           "null_right": AlgebraOp("null", add_nothing),
           "null_left": AlgebraOp("null", add_nothing),
           "collapse": AlgebraOp("collapse", lambda unary_list: collapse(unary_list[0]))
           }

tag_algebra = Algebra(ops=tag_ops, name="TAG string algebra")
tag_algebra.add_constant_maker(lambda word: AlgebraOp(repr(make_right_pair(word)), make_right_pair(word)))

# lambda string, left=False: AlgebraOp(make_pair(string, left).__repr__(), make_pair(string, left))

# hello = tag_algebra.constant_maker("hello")
#
# print(hello)
#
# print(make_left_pair("hello"))
#
# t = Tree("lr", [Tree("hello"), Tree("world")])
#
# term = tree2term(t, tag_algebra)
#
# print(term.evaluate())