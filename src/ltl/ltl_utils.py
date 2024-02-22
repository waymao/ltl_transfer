from typing import Tuple, Union

LTL = Union[
    str,
    Tuple[str, 'LTL', 'LTL'],
    Tuple[str, 'LTL']
]

DFAEdge = Tuple[str, str]


# from copilot
def convert_ltl(a: LTL) -> str:
    if isinstance(a, str):
        return a
    elif isinstance(a, list):
        if a[0] == "and":
            return " & ".join([convert_ltl(a_) for a_ in a[1:]])
        elif a[0] == "until":
            if a[1] == "True":
                return "F(" + convert_ltl(a[2]) + ")"
            else:
                return "U(" + convert_ltl(a[1]) + ", " + convert_ltl(a[2]) + ")"
        elif a[0] == "next":
            return "X(" + convert_ltl(a[1]) + ")"
        elif a[0] == "True":
            return "True"
        elif a[0] == "not":
            return "!" + convert_ltl(a[1])
        else:
            print(a)
            raise NotImplementedError
    else:
        raise NotImplementedError
