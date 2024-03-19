from ..minimalism.movers import Slot

A = "A"  # A1 from Torr thesis
Abar = "ABar"  # A'1
R = "R"  # A'2
Self = "Self"  # A2
E = "E"  # for ListMovers; we can add as many as we want to this list


def slot_name2slot(name):
    """
    The slots used in converting from MGBank
    A yields an A-slot
    R, Self, and Abar yield normal slots
    E yields a multiple slot
    everything else (expecting just epp, num, pers, and loc) yields an epp slot
    @param name: str
    @return: Slot
    """
    if name == A:
        return Slot(A, a=True)
    elif name in {Abar, R, Self}:
        return Slot(name)
    elif name == E:
        return Slot(E, multiple=True)
    else:
        return Slot(name, epp=True)



