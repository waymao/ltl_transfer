# from copilot
def ltl_to_print(a):
    if isinstance(a, str):
        return a
    elif isinstance(a, list) or isinstance(a, tuple):
        if a[0] == "and":
            return " & ".join([ltl_to_print(a_) for a_ in a[1:]])
        elif a[0] == "until":
            if a[1] == "True":
                return "F(" + ltl_to_print(a[2]) + ")"
            else:
                return "(" + ltl_to_print(a[1]) + " U " + ltl_to_print(a[2]) + ")"
        elif a[0] == "next":
            return "X(" + ltl_to_print(a[1]) + ")"
        elif a[0] == "not":
            return "!" + ltl_to_print(a[1])
        elif a[0] == "True":
            return "True"
        else:
            raise NotImplementedError("Cannot format {}".format(a))
    else:
        raise NotImplementedError("Cannot format {}".format(a))
