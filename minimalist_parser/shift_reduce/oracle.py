"""
Functions for making an oracle out of derivation trees
"""

from minimalist_parser.shift_reduce import Expr, Feature


def make_dt(tree):
    return DerivationTree(tree.parent, tree.children)


class DerivationTree:
    def __init__(self, parent=None, children=None):
        self.label = parent
        self.children = children

    def __repr__(self):
        s = self.label
        if self.children is not None:
            s += "("
            child = self.children[0]
            s += child.__repr__
            for child in self.children[1:]:
                s += ", "
                s += child.__repr__
            s += ")"
        return s

    def interpret(self):
        """
        Interprets a derivation tree into a minimalist string algebra
        @return: Expr
        """
        if self.children is None:  # leaf
            return Expr(self.label)  # make an expression out of the label
        # operations: interpret daughters, apply operation to them
        elif self.label == "mgR":
            assert len(self.children) == 2
            expr = self.children[0].interpret()
            expr.concat_right(self.children[1].interpret())
            return expr
        elif self.label == "mgL":
            assert len(self.children) == 2
            expr = self.children[0].interpret()
            expr.concat_left(self.children[1].interpret())
            return expr
        elif self.label.startswith("mg_"):  # eg mg_wh
            parts = self.label.split("_")
            f = parts[1]  # eg wh
            assert len(self.children) == 2
            expr = self.children[0].interpret()
            expr.merge_2(self.children[1].interpret(), Feature(f))
            return expr
        elif self.label.startswith("mv"):
            parts = self.label.split("_")
            f = parts[1]
            assert len(self.children) == 1
            if len(parts) == 2:  # mv_1
                expr = self.children[0].interpret()
                expr.move_left(Feature(f))
                return expr
            elif len(parts) == 3:  # mv_2
                g = parts[2]  # second feature
                expr = self.children[0].interpret()
                expr.move_2(Feature(f), Feature(g))
                return expr
            else:
                print("weird Move node label")
                exit()
        else:
            print("weird node label")
            exit()

    def make_high_mover_tree(self):
        """

        @return:
        """
        if self.children is None:
            return Expr(self, make_dt)
        elif self.label == "mgR":
            pass
        elif self.label.startswith("mg_"):  # eg mg_wh
            parts = self.label.split("_")
            f = parts[1]  # eg wh
            assert len(self.children) == 2
            expr = self.children[0].make_high_mover_tree()
            expr.merge_2(self.children[1].make_high_mover_tree(), Feature(f))
            return expr
        elif self.label.startswith("mv"):
            parts = self.label.split("_")
            f = parts[1]
            assert len(self.children) == 1
            if len(parts) == 2:  # mv_1
                expr = self.children[0].make_high_mover_tree()
                expr.move_left(Feature(f))
                return expr
            elif len(parts) == 3:  # mv_2
                return Expr(self, make_dt)
            else:
                print("weird Move node label")
                exit()
        else:
            pass





def order(tree):
    if tree.parent == "mgL":
        tree.children = tree.children.reverse()


class DerTreeHighMovers(DerivationTree):
    def __init__(self, parent=None, children=None):
        super().__init__(self, parent, children)
        self.movers = {}


    def movers_up(self, tree):
        if tree.parent.startswith("mg_"):
            parts = self.label.split("_")
            f = parts[1]  # eg wh
            self.movers[f] = self.movers_up(tree.children[1])
            self.label = tree.parent
            self.children = self.movers_up(tree.children[0])
            return self
        elif tree.parent.startswith("mv"):
            parts = self.label.split("_")
            f = parts[1]
            assert len(self.children) == 1
            if len(parts) == 2:  # mv_1
                mover = self.movers.pop(f)
                return
            elif len(parts) == 3:  # mv_2
                g = parts[2]  # second feature
                expr = self.children[0].interpret()
                expr.move_2(Feature(f), Feature(g))
                return expr






# example

t = DerivationTree("mv_wh",
                   [DerivationTree("mv_k_wh",
                                   [DerivationTree("mg_k", [DerivationTree("slept"), DerivationTree("who")])])])

print(t)

print(t.interpret())

