"""
implements a typeless version of the AM algebra. We only care about whether we can apply the operation, not whether we should
"""
import logging
from copy import deepcopy

from overrides import overrides

from .algebra import *
from .algebra_objects.graphs import SGraph

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class AMAlgebra(Algebra):
    """
    A typeless AM algebra.
    Apply and Modify work as long as the underlying operations work, otherwise they throw the underlying errors.
    Domain: SGraphs.
    Binary operations: Apply and Modify (by source).
    Attributes:
        zero: empty graph.
        domain_type: the type SGraph.
        sources: list of strings (could be anything) for the source of the SGraphs.
        app_only: sources that are only used in APP.
        mod_only: sources that are only used in MOD.
        ops: dict of operation names to AlgebraOps. Traditionally just the non-constants
        constant_maker: a function that makes constant AlgebraOps.
         For SGraphs you probably want to add this after initialisation by adding a dict of constants,
         which makes the constant maker a look-up function.

    """

    def __init__(self, sources=None, app_only=None, mod_only=None, graph_type=SGraph):
        super().__init__(name="AM algebra", domain_type=graph_type)
        self.add_constant_maker()
        print("added constant maker")

        self.OP = "OP"

        # some AMR defaults
        if sources is None:
            sources = ["M", "S", "O"]
            for i in range(1, 8):
                sources.append(f"{self.OP}{i}")
            if self.domain_type == SGraph:
                sources.append(self.OP)
            else:
                for i in range(1, 20):
                    sources.append(f"{self.OP}{i}")
            if app_only is None:
                app_only = [source for source in sources if source.startswith(self.OP)]
            if mod_only is None:
                mod_only = ["M"]

        self.sources = sources
        self.app_only = app_only
        self.mod_only = mod_only

        self.ops = {}

        #print(self.sources)
        #print(self.mod_only)

        # all Algebra operations defined here
        for s in [x for x in self.sources if x not in self.mod_only]:
            function = self.make_apply_operation(s)
            name = f"App_{s}"
            op = AlgebraOp(name, function)
            self.ops[op.name] = op
        for s in [x for x in self.sources if x not in self.app_only]:
            function = self.make_modify_operation(s)
            name = f"Mod_{s}"
            op = AlgebraOp(name, function)
            self.ops[op.name] = op
        if self.domain_type == SGraph:
            name = "Add_conjunct"
            self.ops[name] = AlgebraOp(name, self.add_conjunct)

    def __repr__(self):
        return self.name

    def ops_repr(self):
        apps = []
        mods = []
        other = []
        for op in self.ops:
            if op[:3].lower() == "app":
                apps.append(op)
            elif op[:3].lower() == "mod":
                mods.append(op)
            else:
                other.append(op)
        ret = "\n\tApp:\n\t\t"
        for op in sorted(apps):
            ret += f" {op}"
        ret += "\n\tMod:\n\t\t"
        for op in sorted(mods):
            ret += f" {op}"
        ret += "\n\tOther:\n\t\t"
        for op in sorted(other):
            ret += f" {op}"
        return ret

    def make_apply_operation(self, source):
        def apply_to_list(args):
            head = args[0]
            argument = args[1]
            # def apply_source(head: SGraph, argument: SGraph):
            return self.apply(source, head, argument)
            # return apply_source

        return apply_to_list

    def make_modify_operation(self, source):
        def modify_source(args):
            head = args[0]
            modifier = args[1]
            return self.modify(source, head, modifier)

        return modify_source

    def make_leaf(self, name, function=None):
        return AlgebraTerm(self.constant_maker(name))

    @overrides
    def default_constant_maker(self, word, label=None):
        """
        Makes unary graph with word as label
        """
        logger.debug("using default constant maker")
        return AlgebraOp(word, self.domain_type({0}, node_labels={0: word}, root=0))

    def add_conjunct(self, args):
        head, conjunct = args[0], args[1]
        try:
            head.add_op_source()
            return self.apply(self.domain_type().OP, head, conjunct)
        except AttributeError:
            raise NotImplementedError

    def apply(self, source: str, head: SGraph, argument: SGraph):
        """
        Implements the AM Apply operation, with no typing.
        @param source: str: the source to put the root of argument into.
        @param head: the head and functor of the operation (the one with the relevant source).
        @param argument: the argument (the one to be added to head).
        @return: SGraph: the result of Apply_source(head, argument).
        """
        logger.debug(f"App{source}\n{head}\n {argument}\n")
        assert isinstance(head, SGraph), f"head must be an SGraph but is a {type(head)}"
        assert isinstance(argument, SGraph), f"argument must be an SGraph but is a {type(head)}"
        assert source in head.sources, f"App_{source} is not possible without {source} in the head. Head: {head}"
        # don't want to copy the whole thing just to add the source, so we add it in place and then remove it
        argument.add_source(argument.root, source)
        logger.debug(f"Updated argument: {argument}")
        new_graph = head + argument
        new_graph.forget(source)
        argument.forget(source)
        return new_graph

    def modify(self, source: str, head: SGraph, modifier: SGraph):
        """
         implements the AM Modify operation, with no typing
         @param source: str: the source to put the root of argument into
         @param head: the graph to have a modifier added to. Will keep this one's root.
         @param modifier: the modifier to be added (functor; i.e. the one with the relevant source)
         @return: SGraph
        """

        logger.debug(f"Mod_{source}  of \n {head} \n {modifier}")
        # use a special temporary source name
        tmp = "TMP"
        assert isinstance(head, SGraph), f"head must be an SGraph but is a {type(head)}"
        assert isinstance(modifier, SGraph), f"modifier must be an SGraph but is a {type(head)}"
        assert tmp not in head.sources and tmp not in modifier.sources
        assert source in modifier.sources, f"Mod_{source} is not possible without {source} in the modifier"

        # We want to merge the root of the head and the given source of the modifier
        # don't want to copy the whole thing just to add the source, so we add it in place and then remove it
        head.add_source(head.root, tmp)
        modifier.rename(source, tmp)

        new_graph = head + modifier
        new_graph.forget(tmp)

        # put it back
        modifier.rename(tmp, source)
        head.forget(tmp)
        return new_graph


if __name__ == "__main__":
    from amr import make_amr_named_entity, make_amr_predicate

    adjunctizer = SGraph(
        {0, 1},
        {1: [(0, "mod")]},
        sources={"M": 1, "X": 0},
        root=0
    )

    whistle = make_amr_predicate("whistle-01", [0])

    and_graph = SGraph(
        {0, 1},
        {
            1: [(0, "op1")]
        },
        {1: "and"},
        {"op1": 0},
        1
    )

    sleep = make_amr_predicate("sleep-01", [0])
    dream = make_amr_predicate("dream-01", [0])

    vocabulary = {"mary": make_amr_named_entity("person", ["Mary"]),
                  "tried": make_amr_predicate("try-01", [0, 1]),
                  "like": make_amr_predicate("like-01", [0, 1]),
                  "enter": make_amr_predicate("enter-01", [0, 1]),
                  "whistle": whistle,
                  "and": and_graph,
                  "sleep": sleep,
                  "slept": sleep,
                  "dream": dream,
                  "dreamt": dream,
                  "cat": make_amr_predicate("cat", []),
                  "I": make_amr_predicate("i", []),
                  "room": make_amr_predicate("room", []),

                  }

    print("Vocabulary:")
    for word in vocabulary:
        print(word, vocabulary[word])
        p = vocabulary[word].to_penman()
        print(p)
        print("top:", p.top)

    # make the algebra
    am_alg = AMAlgebra(graph_type=SGraph)
    am_alg.add_constants({name: AlgebraOp(name, vocabulary[name]) for name in vocabulary})

    # examples
    print("AppO(enter, room)")
    enter_room = am_alg.apply("O", vocabulary["enter"], vocabulary["room"])
    print("result:", enter_room)

    print("ModX(whistle, [adjunctizer]")
    whistling = am_alg.modify("X", whistle, adjunctizer)
    print(whistling)

    print("ModM(enter the room, whistling)")
    enter_room_whistling = am_alg.modify("M", enter_room, whistling)
    print(enter_room_whistling)

    print("APPS(enter the cat whistling, I)")
    full_amr = am_alg.apply("S", enter_room_whistling, vocabulary["I"])
    print(full_amr)

    # Tests: use to_penman method to take advantage of the isomorphism check they have
    # to make sure students are getting the right outputs.
    # also check that vocabulary isn't changed in place.


if __name__ == "__main__":
    from amr import make_amr_named_entity, make_amr_predicate

    adjunctizer = SGraph(
        {0, 1},
        {1: [(0, "mod")]},
        sources={"M": 1, "X": 0},
        root=0
    )

    whistle = make_amr_predicate("whistle-01", [0])

    and_graph = SGraph(
        {0, 1},
        {
            1: [(0, "op1")]
        },
        {1: "and"},
        {"op1": 0},
        1
    )

    sleep = make_amr_predicate("sleep-01", [0])
    dream = make_amr_predicate("dream-01", [0])

    vocabulary = {"mary": make_amr_named_entity("person", ["Mary"]),
                  "tried": make_amr_predicate("try-01", [0, 1]),
                  "like": make_amr_predicate("like-01", [0, 1]),
                  "enter": make_amr_predicate("enter-01", [0, 1]),
                  "whistle": whistle,
                  "and": and_graph,
                  "sleep": sleep,
                  "slept": sleep,
                  "dream": dream,
                  "dreamt": dream,
                  "cat": make_amr_predicate("cat", []),
                  "I": make_amr_predicate("i", []),
                  "room": make_amr_predicate("room", []),

                  }

    print("Vocabulary:")
    for word in vocabulary:
        print(word, vocabulary[word])
        p = vocabulary[word].to_penman()
        print(p)
        print("top:", p.top)

    # make the algebra
    am_alg = AMAlgebra(graph_type=SGraph)
    am_alg.add_constants({name: AlgebraOp(name, vocabulary[name]) for name in vocabulary})

    # examples
    print("AppO(enter, room)")
    enter_room = am_alg.apply("O", vocabulary["enter"], vocabulary["room"])
    print("result:", enter_room)

    print("ModX(whistle, [adjunctizer]")
    whistling = am_alg.modify("X", whistle, adjunctizer)
    print(whistling)

    print("ModM(enter the room, whistling)")
    enter_room_whistling = am_alg.modify("M", enter_room, whistling)
    print(enter_room_whistling)

    print("APPS(enter the cat whistling, I)")
    full_amr = am_alg.apply("S", enter_room_whistling, vocabulary["I"])
    print(full_amr)
