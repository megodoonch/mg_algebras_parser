from abc import ABC


class PreparePackages(ABC):
    def __init__(self, name, inner_algebra):
        self.name = name
        self.inner_algebra = inner_algebra

    def __repr__(self):
        return self.name

