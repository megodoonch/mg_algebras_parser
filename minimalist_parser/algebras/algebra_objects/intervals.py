from ..hm_algebra import HMError

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
log = logger.debug

class IntervalError(HMError):
    """Exception raised for errors in concatenation attempts and subtraction
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = "IntervalError: " + message


class Interval:
    """
    Interval of natural numbers
    We index between the words, so words have length 1
    Note: empty interval is (None, None) of length 0

    Attributes:
        start: int or None; left end of the interval
        end: int or None; right end of the interval

    """

    def __init__(self, start=None, length=1):
        """
        Interval (start, start + length)
        Note: empty interval is (None, None) of length 0
        @param start: int or None
        @param length: int
        """
        if start is not None and start < 0 or length < 0:
            raise IntervalError(f"invalid interval starting at {start} with length {length}")
        self.start = start  # None for empty interval
        if start is None:
            self.end = None
        else:
            self.end = start + length

    def __len__(self):
        return 0 if self.start is None else self.end - self.start

    def __repr__(self):
        """
        If you update this, update string2self too
        @return:
        """
        string = self.start
        e = self.end
        if len(self) == 0:
            string = "_"
            e = "_"
        return "({},{})".format(string, e)

    @staticmethod
    def string2self(string):
        """
        Inverse of __repr__
        @param string:
        @return:
        """
        print(string)
        split_string = string.split(",")
        start = split_string[0][1:]
        end = split_string[1][:-1]
        if start == "_":  # empty interval
            return Interval()
        start = int(start)
        end = int(end)
        return Interval(start=start, length=end - start)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.start == other.start and self.end == other.end

    def __lt__(self, other):
        """
        self = (a,b) < other = (c,d) iff b <= c
         (whole interval precedes whole interval)
        @param other: Interval
        @return: True if whole self interval is before whole other interval
        """
        return len(other) == 0 or len(self) == 0 or self.end <= other.start

    def __gt__(self, other):
        """
        self = (a,b) > other = (c,d) iff b >= c
        (whole interval follows whole interval)
        @param other: Interval
        @return: True if whole self interval is after whole other interval
        """
        return len(other) == 0 or len(self) == 0 or self.start >= other.end

    def __add__(self, other):
        """
        Adds two adjacent intervals (a,b)+(b,c) = (a,c)
        @param other: Interval
        @return: Interval
        """
        # print("adding {} and {}".format(self, other))
        if len(self) == 0:
            return other
        elif len(other) == 0:
            return self
        if self.end == other.start:
            return Interval(self.start, len(self) + len(other))
        else:
            raise IntervalError(f"can't add {self} + {other}")

    def __sub__(self, other):
        """
        returns absolute distance between two intervals, and raises
         IntervalError if they overlap
        @param other: Interval
        @return: int
        """
        if len(self) == 0 or len(other) == 0:  # distance is 0 to silent things
            return 0
        elif self.__lt__(other):
            return other.start - self.end
        elif self.__gt__(other):
            return self.start - other.end
        else:
            raise IntervalError("intervals overlap")

    def concat_commute(self, other):
        """
        Commutative addition of self with another interval
        @param other: Interval
        @return: Interval
        """
        try:
            return self + other
        except IntervalError:
            try:
                return other + self
            except IntervalError:
                raise IntervalError(f"{self} and {other} not adjacent")

    # @staticmethod
    # def concat(intervals):
    #     ints = Interval(length=0)
    #     if len(intervals) == 0:
    #         return ints
    #     else:
    #         for interval in intervals[1:]:
    #             ints += interval
    #     return ints

    def collapse(self):
        """
        Dumb hack for compatability with Pair
        @return:
        """
        return self

    def is_immediately_left_of(self, other):
        """
        true iff self is right before other, or either is empty
        @param other: Interval
        @return: bool
        """
        assert type(other) == Interval, f"{other} should be an Interval, but it's a {type(other)}"
        return self.end == other.start or len(self) == 0 or len(other) == 0

    @staticmethod
    def empty_interval():
        return Interval()


class Pair:
    def __init__(self, first_interval: Interval, second_interval: Interval):  # , hm=False):
        """
        @param first_interval: head's interval.
        @param second_interval: Interval of the rest.
        """
        # this will raise IntervalError if intervals overlap
        first_interval - second_interval

        self.head = first_interval
        self.rest = second_interval

        # if self.gap():
        #     self.must_hm = True
        # else:
        #     self.must_hm = hm

    def __repr__(self):
        """
        if you update this, update string2self too.
        @return:
        """
        return "<{}, {}>".format(self.head, self.rest)

    def __eq__(self, other):
        return isinstance(other,
                          Pair) and self.head == other.head and self.rest == other.rest  # and self.must_hm == other.must_hm

    @staticmethod
    def string2self(string: str):
        parts = string.split(", ")
        first = parts[0][1:]  # delete "<"
        second = parts[1][:-1]  # delete ">"
        head = Interval.string2self(first)
        rest = Interval.string2self(second)
        return Pair(first_interval=head, second_interval=rest)

    def __len__(self):
        return len(self.head) + len(self.rest) \
            + (self.rest - self.head)

    def collapse(self) -> Interval:
        """
        Try collapsing the pair
        @return: new Interval with head and rest concatenated
                None if must_hm or intervals aren't adjacent
        """
        # if self.must_hm:
        #     raise HMError(f"Can't collapse {self} b/c head must move")
        if self.gap():
            raise HMError(f"Can't collapse {self} because pair is non-adjacent")
        interval = self.head.concat_commute(self.rest)
        assert type(interval) == Interval, f"{interval} should be an Interval, but it is a {type(interval)}"
        return interval

    def gap(self):
        """
        true if there's a gap
        @return: bool
        """
        try:
            self.head.concat_commute(self.rest)
            return False
        except IntervalError:
            return True


class PairItem:
    """
    Objects for Interval Pair Algebra.
    items are either Intervals or Pairs, and are typed with whether head movement must apply.
        This can't be determined from the item otherwise.
    Attributes:
        span: the interval or pair.
        head: which part of the span is the head.
        rest: which part of the span isn't the head.
        must_hm: if True, HM must apply.
        tupled: True if type of span is Pair.
        can_hm: if true, HM may apply.
    """

    def __init__(self, interval_or_pair=None, lexical=True, must_hm=False):
        """
        Creates an Item instance with span an interval or pair of intervals,
        and type MGType(lexical, must_hm, conj)
        @param interval_or_pair: Interval or Pair
        @param lexical: boolean
        """
        if interval_or_pair is None:
            self.span = Interval()
        elif isinstance(interval_or_pair, int):
            self.span = Interval(interval_or_pair)
        else:
            self.span = interval_or_pair

        if isinstance(self.span, Pair):
            self.must_hm = must_hm
            self.head = self.span.head
            self.rest = self.span.rest
        elif isinstance(self.span, Interval):
            self.must_hm = False
            if lexical:
                self.head = self.span
                self.rest = Interval()
            else:
                self.head = Interval()
                self.rest = self.span

        else:
            raise TypeError(f"Item must be made from an Interval or Pair,"
                            f" not {type(interval_or_pair)}.")

        self.tupled = isinstance(self.span, Pair)
        self.lexical = lexical
        # HM is possible in all tupled items and if it's just a lexical head
        self.can_hm = self.tupled or lexical

    def __eq__(self, other):
        return isinstance(other, PairItem) and self.span == other.span and self.must_hm == other.must_hm

    def __repr__(self):
        ret = f"{self.span}"
        if self.must_hm:
            ret += ":h"
        return ret

    def __len__(self):
        return len(self.span)

    def collapse(self):
        if self.must_hm:
            raise HMError(f"Can't collapse {self} b/c head must move")
        else:
            return self.span.collapse()

    def spellout(self):
        return self.collapse()

    def concat_suffix(self, head):
        """
        concatenate an Interval to the right of the head of the PairItem
        @param head: Interval
        @return: PairItem
        """
        return PairItem(Pair(self.head + head, Interval()), lexical=False, must_hm=False)

    def concat_prefix(self, head):
        """
        concatenate an Interval to the left of the head of the PairItem
        @param head: Interval
        @return: PairItem
        """
        return PairItem(Pair(head + self.head, Interval()), lexical=False, must_hm=False)
