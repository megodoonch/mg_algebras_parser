"""
string triples (l, h, r)
"""
from ... import logging
import sys

from ..algebra import AlgebraError
from ..hm_algebra import HMError

# logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
# make log() into a logging function (for simplicity)
log = logger.debug

# use this for head movement components
head_separator = " "
# use this for regular merge
word_separator = " "


def concat_strings(list_of_strings: [str]):
    """
    Use a space to concatenate strings for normal merging
    @param list_of_strings:
    @return:
    """
    return word_separator.join([x for x in list_of_strings if len(x) > 0])


def concat_morphemes(m1, m2):
    """
    use a space to concatenate strings during head movement
    @param m1:
    @param m2:
    @return:
    """
    if len(m1) == 0:
        return m2
    if len(m2) == 0:
        return m1
    return m1 + head_separator + m2


class Triple:
    def __init__(self, h=None, r=None, l=None, component_type=None):
        """
        @param h: head
        @param r: rightward material
        @param l: leftward material
        @param component_type: type of the three components. Default str for (str, str, str) triples.
        """
        if h is None and (r is not None or l is not None):
            raise AlgebraError("Triples must have a head (or be empty)")

        # Find the component type from the input or the type of the head, right, or left
        if h is None and component_type is None:
            # no info given: use default str
            self.component_type = str  # default
        elif component_type is not None:
            # prioritise explicit type given
            self.component_type = component_type
        else:
            # otherwise, use the type of the head
            self.component_type = type(h)

        if h is not None and (r is not None or l is not None) and (not isinstance(r, self.component_type) or not isinstance(l, self.component_type)):
            raise AlgebraError(f"component type is {self.component_type.__name__}, but r is {type(r).__name__}, and l is {type(l).__name__}")

        # store the head, right, and left.
        try:
            # If the given type matches the type of the given head, use that directly.
            if isinstance(h, self.component_type):
                self.head = self.component_type() if h is None else h
                self.right = self.component_type() if r is None else r
                self.left = self.component_type() if l is None else l
            # Otherwise, give the given head to the given type
            # for example, Triple("hi", component_type=list) will make ([],["hi"],[])
            else:
                self.head = self.component_type() if h is None else self.component_type(h)
                self.right = self.component_type() if r is None else self.component_type(r)
                self.left = self.component_type() if l is None else self.component_type(l)
        except TypeError:
            logger.warning(
                f"{component_type}() can't be built without an argument, but no head, right, or left was provided. All are set to None.")
            self.head = None
            self.left = None
            self.right = None

        logger.debug(f"built {self} with component type {self.component_type}")

    def __repr__(self):
        return "<{}, {}, {}>".format(self.left, self.head, self.right)

    def __len__(self):
        try:
            return len(self.head) + len(self.right) + len(self.left)
        except TypeError:
            raise AlgebraError(
                f"Triples over {self.component_type} have no length because {self.component_type}s have no length")

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.head == other.head and self.left == other.left and self.right == other.right

    def __add__(self, other):
        """concat right"""
        return type(self)(h=self.head, l=self.left, r=self.concat(self.right, other.collapse()))

    def collapse(self):
        """
        Try collapsing the triple
        @return: new string with head and rest concatenated
        """
        logger.debug(f"Collapsing {self} of component type {self.component_type}")
        if self.component_type == str:
            return concat_strings([self.left, self.head, self.right])
        else:
            return self.left + self.head + self.right

    def concat(self, a, b):
        """
        Concatenate two instances of the component type, e.g. if list, concatenate a and b, or if str, add spaces.
        @param a: self.component_type instance, e.g. str.
        @param b: self.component_type instance.
        @return: self.component_type instance.
        """
        assert isinstance(a, self.component_type) and isinstance(b, self.component_type), f"{a} and/or {b} are not {self.component_type}"
        if self.component_type == str:
            return concat_strings([a, b])
        else:
            return a + b

    def concat_morphemes(self, a, b):
        """
        Sometimes we might want to concatenate morphemes differently from words.
        @param a: self.component_type instance, e.g. str.
        @param b: self.component_type instance.
        @return: self.component_type instance.
        """
        if self.component_type == str:
            return concat_morphemes(a, b)
        else:
            return a + b

    @staticmethod
    def string2self(string):
        """
        creates a Triple, given a string in the form <l, h, r> or s.
        @param string:
        @return: Triple
        """
        parts = string.split(", ")
        if len(parts) == 3:
            first = parts[0][1:]  # delete "<"
            third = parts[2][:-1]  # delete ">"
            return Triple(parts[1], r=third, l=first)
        elif len(parts) == 1:
            return Triple(string)
        else:
            raise HMError(f"couldn't create triple from string {string}: didn't split into 1 or 3")

    def spellout(self):
        """
        Creates a single instance of component_type using collapse.
        (l,h,r) -> lhr
        @return: self.component_type instance
        """
        return self.collapse()

    def concat_left(self, other):
        """
        concatenate other to the left of self.left.
        @param other: Triple.
        @return: Triple.
        """
        return type(self)(h=self.head, l=self.concat(other.collapse(), self.left), r=self.right)

    def concat_suffix(self, suf):
        """
        Concatenates the given suffix to the right of the head.
        @param suf: self.component_type (e.g. str)
        @return: Triple
        """
        return type(self)(h=self.concat_morphemes(self.head, suf), l=self.left, r=self.right)

    def concat_prefix(self, pre):
        """
        Concatenates the given prefix to the left of the head.
        @param pre: self.component_type (e.g. str)
        @return: Triple
        """
        return type(self)(h=self.concat_morphemes(pre, self.head), l=self.left, r=self.right)
