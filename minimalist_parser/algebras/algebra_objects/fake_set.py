from typing import List


class FakeSet(List):
    """
    A set-like list object.
    Equality ignores order.
    + is defined as set-formation, so {1} + {2} = {{1}, {2}}
    spellout is defined for ease of MG usage.
    """

    def __init__(self, items=None):
        if items is None:
            items = []
        super().__init__(sorted(items))

    def spellout(self):
        for element in self:
            if isinstance(element, FakeSet):
                element.sort()
        self.sort()
        return self

    def __add__(self, other):
        return FakeSet(sorted([self, other]))

    def __lt__(self, other):
        if isinstance(other, FakeSet):
            for i in range(len(self)):
                if len(other) < i + 1:
                    return False
                if isinstance(self[i], FakeSet) and isinstance(other[i], FakeSet):
                    if self[i] < other[i]:
                        return True
                    elif other[i] < self[i]:
                        return False
                    else:
                        continue
                elif isinstance(self[i], FakeSet) and not isinstance(other[i], FakeSet):
                    return False
                elif not isinstance(self[i], FakeSet) and isinstance(other[i], FakeSet):
                    return True
                else:
                    return self[i] < other[i]

    def __eq__(self, other):
        return isinstance(other, FakeSet) and sorted(self) == sorted(other)

    def __repr__(self):
        return f"{{{', '.join([str(x) for x in self])}}}"
