"""
Implements an MG with features
"""

from collections import defaultdict
# from algebras.possibly_obsolete_algebras import generalised_minimalist_algebra as mga
import minimalist_algebra as mga


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class ExistenceError(Error):
    """Exception raised for errors in the input.

    Attributes:

        message -- explanation of the error
    """

    def __init__(self, message):

        self.message = message


polarity_strings = {
            (1, "sel", None) : "=",
            (-1, "sel", None): "",
            (1, "lic", None) : "+",
            (-1, "lic", None): "-",
        }


class FeatureClass:
    """
    everything about a feature except its name
    """

    def __init__(self, polarity: int, feature_set: str, subclass=None):
        """
        @param polarity: +1 or -1 for positive (=, +) and negative (cat, -)
        @param feature_set: str: sel or lic
        @param subclass: for e.g. covert move, left merge
        """
        if polarity not in [1, -1]:
            raise ExistenceError("polarity must be 1 or -1")
        self.polarity = polarity
        if feature_set not in ["sel", "lic"]:
            raise ExistenceError("feature_set must be lic or sel")
        self.feature_set = feature_set
        self.subclass = subclass

    def __repr__(self):
        if (self.polarity, self.feature_set, self.subclass) in polarity_strings:
            return polarity_strings[(self.polarity, self.feature_set, self.subclass)]
        else:
            return "?"

    def __eq__(self, other):
        return self.feature_set == other.feature_set and self.subclass == other.subclass\
            and self.polarity == other.polarity

    def __hash__(self):
        return hash((self.polarity, self.feature_set, self.subclass))


class Feature:
    """

    """
    def __init__(self, label: str, feature_class: FeatureClass):
        """
        MG feature class
        @param label: str; name, eg N or wh

        """

        self.feature_class = feature_class
        self.label = label

    def __repr__(self):
        """
        Concatenate the polarity marker and the label, eg -f or =N
        @return: str
        """
        return f"{self.feature_class}{self.label}"

    def __eq__(self, other):
        """
        Equal if all your parts are equal
        @param other:
        @return: bool
        """
        return self.label == other.label and self.feature_class == other.feature_class

    def __hash__(self):
        """
        Hash on the string representation
        @return: bool
        """
        return hash(repr(self))

    def __lt__(self, other):
        """
        Alphabetical order
        @param other:
        @return:
        """
        return self.label < other.label


class LexicalItem:
    """
    MG lexical item with label and feature stack
    """
    def __init__(self, label: str, features: [Feature]):
        """
        Creates lexical item with given string and feature stack
        @param label: str
        @param features: list of Feature objects
        """
        self.label = label
        self.features = features

    def __repr__(self):
        return f"({self.label}, {self.features})"


class MG:
    """
    Attributes:
        lexicon: set of LexicalItems
        features: set of all features used in lexicon
    """
    def __init__(self, lexicon: {LexicalItem}, algebra: mga.MinimalistAlgebra, polarity2algebra_ops):
        """
        Stores the lexicon and extracts the features
        @param lexicon:
        @param polarity2algebra_ops: TODO this is wrong: MG2 ops can depend on both selector and mover, eg HM + store
        """
        self.lexicon = lexicon
        self.polarity2algebra_ops = polarity2algebra_ops
        self.algebra = algebra

        # flat set of features in the lexicon
        self.features = set([feature for features in
                         [li.features for li in self.lexicon]
                         for feature in features])
        # set of feature classes in the lexicon
        self.feature_classes = set([f.feature_class for f in self.features])

        # divided into selectional and licensing
        self.selectional_features = [f for f in self.features if f.feature_class.feature_set == "sel"]
        self.licensing_features = [f for f in self.features if f.feature_class.feature_set == "lic"]

        # extract the consecutive Cat Neg and Neg Neg pairs for MG2 and MV2
        self.cat_negs = set()
        self.neg_negs = set()
        for li in self.lexicon:
            for i, f in enumerate(li.features):
                if f.feature_class.feature_set == "lic" and f.feature_class.polarity == -1:
                    if li.features[i - 1].feature_class.feature_set == "sel":
                        self.cat_negs.add((li.features[i - 1].feature_class, li.features[i]))
                    else:
                        self.neg_negs.add((li.features[i - 1], li.features[i]))

        self.merge1 = defaultdict(set)
        self.merge_1_ops()
        # self.merge_2_ops()

    def __repr__(self):
        s = "Features:"
        s += f"\n\tsel: {sorted([f for f in self.selectional_features if f.feature_class.polarity == 1])}"
        s += f"\n\t   : {sorted([f for f in self.selectional_features if f.feature_class.polarity == -1])}"
        s += f"\n\tlic: {sorted([f for f in self.licensing_features if f.feature_class.polarity == 1])}"
        s += f"\n\t   : {sorted([f for f in self.licensing_features if f.feature_class.polarity == -1])}"

        s += "\n\nAlgebra:"
        for polarity in self.polarity2algebra_ops:
            ops = self.polarity2algebra_ops[polarity]
            s += f"\n{polarity}:"
            s += f"{ops}"

        return s

    def merge_1_ops(self):

        # go through polarities actually in the lexicon
        for f in [f for f in self.selectional_features if f.feature_class.polarity == 1]:
            self.merge1[f.feature_class] = self.polarity2algebra_ops["MG1"][f.feature_class]

    # def merge_2_ops(self):
    #     for (selector, licensee) in self.cat_negs:
    #         self.merge_2[(selector,licensee)] = self.algebra.mg2






if __name__ == "__main__":

    # basic feature classes
    cat = FeatureClass(-1, "sel")
    sel = FeatureClass(1, "sel")
    pos = FeatureClass(1, "lic")
    neg = FeatureClass(-1, "lic")

    # some negative features
    # sel
    n = Feature("N", cat)
    d = Feature("D", cat)
    v = Feature("V", cat)
    t = Feature("T", cat)
    c = Feature("R", cat)

    # lic
    wh = Feature("wh", neg)
    k = Feature("k", neg)

    def select(c: Feature):
        return Feature(c.label, sel)

    def license(f: Feature):
        return Feature(f.label, pos)

    # some basic feature stacks

    # nominal hierarchy
    noun = [n]
    det = [Feature("N", sel), d]
    wh_det = [Feature("N", sel), d, wh]
    subj = [Feature("N", sel), d, k]
    wh_subj = [Feature("N", sel), d, k, wh]


    # verbal hierarchy
    intr = [select(d), v]
    tr = [select(d), select(d), v]
    t_feats = [select(v), license(k), t]
    c_wh_feats = [select(t), license(wh), c]


    # lexicon
    which = LexicalItem("which", wh_det)
    which_subj = LexicalItem("which_subj", wh_subj)
    squidgyhead = LexicalItem("squidgyhead", noun)
    saw = LexicalItem("saw", tr)
    woogled = LexicalItem("woogled", intr)
    tense = LexicalItem("tense", t_feats)
    c_wh = LexicalItem("C_wh", c_wh_feats)

    string_algebra = mga.algebras["string"]
    algebra = mga.MinimalistAlgebra(string_algebra["inner_alg"])
    algebra.add_all_ops(string_algebra["ops"])

    mapping = {
        "MG1": {
            sel: algebra.mg1["mg_lr"],
        },
        "MG2": {
            (sel, neg): algebra.mg2
        },
        "MV1": {
            pos: algebra.mv1
        },
        "MV2": {
            (pos, neg): algebra.mv2
        }
    }

    g = MG({which, which_subj, squidgyhead, saw, woogled, tense, c_wh}, algebra, mapping)

    # print(g)
